"""Run each web synthesis request with its installed model's Python environment."""

from __future__ import annotations

from collections.abc import Callable
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

if TYPE_CHECKING:
    from tts_audiobook_tool.web_backend import BookSession
    from tts_audiobook_tool.web_render import RenderVoice


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
    cancel_event: Event | None = None,
) -> str:
    return _run(
        model_id,
        dict(action="sample", text=text, voice=asdict(voice), language_code=language_code),
        cancel_event=cancel_event,
    )


def transcribe_model_reference(path: str, *, language_code: str = "en") -> str:
    """Run the cached local speech recognizer without using GPU memory."""
    return _run(None, dict(action="transcribe", path=path, language_code=language_code))


def render_model_book(
    model_id: str, session: BookSession, voices: dict[str, RenderVoice], *,
    on_progress: Callable[[int, int], None] | None = None, cancel_event: Event | None = None,
    validate_audio: bool = True, batch_size: int = 1,
) -> str:
    from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict

    return _run(
        model_id,
        dict(
            action="book",
            book=book_to_project_text_json_dict(session.book),
            project_dir=session.project_dir,
            roster=session.roster.to_dict() if session.roster else None,
            attributions=session.attributions.to_dict(),
            voices={name: asdict(voice) for name, voice in voices.items()},
            validate_audio=validate_audio,
            batch_size=batch_size,
        ),
        on_progress=on_progress,
        cancel_event=cancel_event,
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
) -> str:
    with _RENDER_LOCK, TemporaryDirectory(prefix="audiobook-render-") as directory:
        if cancel_event is not None and cancel_event.is_set():
            raise ModelRunError("Generation stopped.")
        work = Path(directory)
        (work / "input.json").write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.Popen(
            [str(_python_for(model_id)) if model_id else sys.executable,
             "-m", "tts_audiobook_tool.web_models", directory],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=(os.name == "posix"),
        )
        previous = None
        try:
            while proc.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    raise ModelRunError("Generation stopped.")
                if on_progress is not None and (work / "progress.json").is_file():
                    progress = json.loads((work / "progress.json").read_text(encoding="utf-8"))
                    if progress != previous:
                        on_progress(*progress)
                        previous = progress
                try:
                    proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    continue
            try:
                result = json.loads((work / "result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise ModelRunError(
                    f"The speech model exited with code {proc.returncode}. Check available GPU memory."
                ) from None
            if "error" in result:
                raise ModelRunError(result["error"])
            if payload["action"] == "transcribe":
                return result["text"]
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
    from tts_audiobook_tool.web_render import RenderVoice, WebRenderError, render_book, render_sample

    payload = json.loads((directory / "input.json").read_text(encoding="utf-8"))
    cancelled = Event()
    signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
    try:
        if payload["action"] == "transcribe":
            from tts_audiobook_tool.web_render import transcribe_reference

            transcript = transcribe_reference(payload["path"], language_code=payload["language_code"])
            _write_json(directory / "result.json", {"text": transcript})
            return
        if payload["action"] == "sample":
            output = render_sample(
                payload["text"], RenderVoice(**payload["voice"]),
                language_code=payload["language_code"], cancel_event=cancelled,
            )
        elif payload["action"] == "book":
            book = book_from_project_text_json_dict(payload["book"])
            if isinstance(book, str):
                raise ModelRunError("Could not read the selected book.")
            session = BookSession(
                book=book, project_dir=payload["project_dir"],
                roster=Roster.from_dict(payload["roster"]) if payload["roster"] else None,
                attributions=AttributionsFile.from_dict(payload["attributions"]),
            )

            def report(current: int, total: int) -> None:
                _write_json(directory / "progress.json", [current, total])

            voices = {name: RenderVoice(**voice) for name, voice in payload["voices"].items()}
            output = render_book(
                session, voices, on_progress=report, cancel_event=cancelled,
                validate_audio=payload["validate_audio"], batch_size=payload["batch_size"],
            )
        else:
            raise ModelRunError("Unknown generation request.")
        result = {"output": output}
    except (WebRenderError, ModelRunError) as error:
        result = {"error": str(error)}
    finally:
        ModelWorker.shutdown()
    _write_json(directory / "result.json", result)


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
