"""Run each web synthesis request with its installed model's Python environment."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from functools import lru_cache
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock
from typing import TYPE_CHECKING

from tts_audiobook_tool.constants_config import PROJECT_MAX_RETRIES_DEFAULT
from tts_audiobook_tool.generation_events import BookRenderTelemetry
from tts_audiobook_tool.web_console import capture_synthesis_console, relay_synthesis_console
from tts_audiobook_tool.web_remote_models import REMOTE_MODEL_ID, WebRemoteTtsError, remote_client_python

if TYPE_CHECKING:
    from tts_audiobook_tool.web_backend import BookSession
    from tts_audiobook_tool.web_remote_models import RemoteTtsSelection
    from tts_audiobook_tool.web_render import RenderVoice
    from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettings


ROOT = Path(__file__).resolve().parents[1]
_RENDER_LOCK = Lock()


class ModelRunError(ValueError):
    """A synthesis error safe to display in the web UI."""


@lru_cache(maxsize=1)
def installed_model_choices() -> list[tuple[str, str]]:
    """Find installed local model environments using the console launcher's probe."""
    from launch import build_venv_list

    return [
        (labels[0], name)
        for _, name, labels in build_venv_list(str(ROOT))
        if len(labels) == 1 and not labels[0].startswith("Remote TTS client")
    ]


def _python_for(model_id: str) -> Path:
    if model_id not in {name for _, name in installed_model_choices()}:
        raise ModelRunError("Select an installed speech model.")
    executable = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return ROOT.joinpath(model_id, *executable)


def render_model_sample(
    model_id: str, text: str, voice: RenderVoice, *, language_code: str = "en",
    cancel_event: Event | None = None, model_options: dict[str, object] | None = None,
    runtime_settings: WebRuntimeSettings | None = None,
    remote_selection: RemoteTtsSelection | None = None,
    on_console: Callable[[str], None] | None = None,
) -> str:
    output = _run(
        model_id,
        dict(
            action="sample", text=text, voice=asdict(voice), language_code=language_code, model_options=model_options,
            runtime_settings=asdict(runtime_settings) if runtime_settings is not None else None,
            remote_selection=asdict(remote_selection) if remote_selection is not None else None,
        ),
        cancel_event=cancel_event,
        on_console=on_console,
    )
    assert output is not None
    return output


def preview_model_reference(path: str, *, trim_start: float = 0.0, trim_end: float | None = None) -> str:
    """Prepare the selected reference audio range without starting a model."""
    from tts_audiobook_tool.web_render import WebRenderError, preview_reference

    try:
        return preview_reference(path, trim_start=trim_start, trim_end=trim_end)
    except WebRenderError as error:
        raise ModelRunError(str(error)) from error


def transcribe_model_reference(
    path: str, *, language_code: str = "en", trim_start: float = 0.0, trim_end: float | None = None,
) -> str:
    """Run the cached local speech recognizer without using GPU memory."""
    transcript = _run(None, dict(
        action="transcribe", path=path, language_code=language_code, trim_start=trim_start, trim_end=trim_end,
    ))
    assert transcript is not None
    return transcript


def render_model_book(
    model_id: str, session: BookSession, voices: dict[str, RenderVoice], *,
    on_progress: Callable[[int, int], None] | None = None, cancel_event: Event | None = None,
    on_telemetry: Callable[[BookRenderTelemetry], None] | None = None,
    validate_audio: bool = True, batch_size: int = 1, model_options: dict[str, object] | None = None,
    first_line: int = 1, last_line: int | None = None, regenerate: bool = False, auto_concat: bool = True,
    strictness: str | None = None, max_retries: int = PROJECT_MAX_RETRIES_DEFAULT,
    runtime_settings: WebRuntimeSettings | None = None,
    remote_selection: RemoteTtsSelection | None = None,
    on_console: Callable[[str], None] | None = None,
) -> str | None:
    from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict

    return _run(
        model_id,
        dict(
            action="book",
            book=book_to_project_text_json_dict(session.book),
            project_dir=session.project_dir,
            roster=session.roster.to_dict() if session.roster else None,
            attributions=session.attributions.to_dict(),
            project_language=session.project_language,
            word_substitutions=session.word_substitutions,
            voices={name: asdict(voice) for name, voice in voices.items()},
            line_voices={slot: asdict(voice) for slot, voice in session.line_voices.items()},
            validate_audio=validate_audio,
            batch_size=batch_size,
            model_options=model_options,
            first_line=first_line,
            last_line=last_line,
            regenerate=regenerate,
            auto_concat=auto_concat,
            strictness=strictness,
            max_retries=max_retries,
            runtime_settings=asdict(runtime_settings) if runtime_settings is not None else None,
            remote_selection=asdict(remote_selection) if remote_selection is not None else None,
        ),
        on_progress=on_progress,
        on_telemetry=on_telemetry,
        cancel_event=cancel_event,
        on_console=on_console,
    )


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait()


def _run(
    model_id: str | None, payload: dict, *,
    on_progress: Callable[[int, int], None] | None = None, cancel_event: Event | None = None,
    on_telemetry: Callable[[BookRenderTelemetry], None] | None = None,
    on_console: Callable[[str], None] | None = None,
) -> str | None:
    with (
        _RENDER_LOCK,
        TemporaryDirectory(prefix="audiobook-render-") as directory,
        capture_synthesis_console(Path(directory), on_console) as (console_options, publish_console),
    ):
        if cancel_event is not None and cancel_event.is_set():
            raise ModelRunError("Generation stopped.")
        remote = payload.get("remote_selection") is not None
        if remote != (model_id == REMOTE_MODEL_ID):
            raise ModelRunError("Choose a verified remote model before selecting remote synthesis.")
        if remote:
            executable = remote_client_python(ROOT)
            if executable is None:
                raise ModelRunError("No installed remote speech client environment was found.")
        else:
            executable = _python_for(model_id) if model_id else sys.executable
        work = Path(directory)
        (work / "input.json").write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.Popen(
            [str(executable), "-m", "tts_audiobook_tool.web_models", directory],
            cwd=ROOT, stdin=subprocess.DEVNULL, start_new_session=(os.name == "posix"),
            **console_options,
        )
        previous = None
        progress_path = work / "progress.json"

        def publish_progress() -> None:
            nonlocal previous
            if on_progress is None and on_telemetry is None:
                return
            try:
                update = BookRenderTelemetry(**json.loads(progress_path.read_text(encoding="utf-8")))
            except FileNotFoundError:
                return
            if update == previous:
                return
            if on_progress is not None and (
                previous is None or (update.processed, update.total) != (previous.processed, previous.total)
            ):
                on_progress(update.processed, update.total)
            if on_telemetry is not None:
                on_telemetry(update)
            previous = update

        try:
            while proc.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    raise ModelRunError("Generation stopped.")
                publish_console()
                publish_progress()
                try:
                    proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    continue
            publish_progress()
            try:
                result = json.loads((work / "result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                if remote:
                    raise ModelRunError(f"The remote speech client exited with code {proc.returncode}.") from None
                raise ModelRunError(
                    f"The speech model exited with code {proc.returncode}. Check available GPU memory."
                ) from None
            if "error" in result:
                raise ModelRunError(result["error"])
            if payload["action"] == "transcribe":
                return result["text"]
            if payload["action"] == "book" and result["output"] is None:
                return None
            output = Path(result["output"])
            if not output.is_file():
                raise ModelRunError("The speech model did not produce an audio file.")
            return str(output)
        finally:
            if proc.poll() is None:
                _stop(proc)


def _write_json(path: Path, data) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _worker(directory: Path) -> None:
    from tts_audiobook_tool import app_support  # Initialize shared app imports first.
    from tts_audiobook_tool.app_types.book_serialization import book_from_project_text_json_dict
    from tts_audiobook_tool.model_worker import ModelWorker
    from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
    from tts_audiobook_tool.speaker_assign.roster import Roster
    from tts_audiobook_tool.web_backend import BookSession
    from tts_audiobook_tool.web_remote_models import RemoteTtsSelection, remote_worker_scope
    from tts_audiobook_tool.web_render import RenderVoice, WebRenderError, render_book, render_sample
    from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettings, WebRuntimeSettingsError

    payload = json.loads((directory / "input.json").read_text(encoding="utf-8"))
    cancelled = Event()
    signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
    scopes = ExitStack()
    remote_options = {}
    try:
        if payload["action"] == "transcribe":
            from tts_audiobook_tool.web_render import transcribe_reference

            transcript = transcribe_reference(
                payload["path"], language_code=payload["language_code"],
                trim_start=payload.get("trim_start", 0.0), trim_end=payload.get("trim_end"),
            )
            _write_json(directory / "result.json", {"text": transcript})
            return
        runtime_settings = WebRuntimeSettings(**(payload.get("runtime_settings") or {}))
        on_console = relay_synthesis_console(directory)
        if payload.get("remote_selection") is not None:
            selection = RemoteTtsSelection(**payload["remote_selection"])
            scopes.enter_context(remote_worker_scope(selection))
            remote_options["remote_selection"] = selection
        if payload["action"] == "sample":
            output = render_sample(
                payload["text"], RenderVoice(**payload["voice"]),
                language_code=payload["language_code"], cancel_event=cancelled,
                model_options=payload.get("model_options"),
                runtime_settings=runtime_settings,
                on_console=on_console,
                **remote_options,
            )
        elif payload["action"] == "book":
            book = book_from_project_text_json_dict(payload["book"])
            if isinstance(book, str):
                raise ModelRunError("Could not read the selected book.")
            session = BookSession(
                book=book, project_dir=payload["project_dir"],
                roster=Roster.from_dict(payload["roster"]) if payload["roster"] else None,
                attributions=AttributionsFile.from_dict(payload["attributions"]),
                project_language=payload.get("project_language"),
                word_substitutions=payload.get("word_substitutions", {}),
                line_voices={int(slot): RenderVoice(**voice) for slot, voice in payload.get("line_voices", {}).items()},
            )

            def report(telemetry: BookRenderTelemetry) -> None:
                _write_json(directory / "progress.json", asdict(telemetry))

            voices = {name: RenderVoice(**voice) for name, voice in payload["voices"].items()}
            output = render_book(
                session, voices, on_telemetry=report, cancel_event=cancelled,
                validate_audio=payload["validate_audio"], batch_size=payload["batch_size"],
                model_options=payload.get("model_options"),
                first_line=payload.get("first_line", 1), last_line=payload.get("last_line"),
                regenerate=payload.get("regenerate", False), auto_concat=payload.get("auto_concat", True),
                strictness=payload.get("strictness"),
                max_retries=payload.get("max_retries", PROJECT_MAX_RETRIES_DEFAULT),
                runtime_settings=runtime_settings,
                on_console=on_console,
                **remote_options,
            )
        else:
            raise ModelRunError("Unknown generation request.")
        result = {"output": output}
    except (WebRenderError, ModelRunError, WebRuntimeSettingsError, WebRemoteTtsError) as error:
        result = {"error": str(error)}
    finally:
        try:
            ModelWorker.shutdown()
        finally:
            scopes.close()
    _write_json(directory / "result.json", result)


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
