"""Browser audio playback from matching takes and the normal model worker."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING

from tts_audiobook_tool.constants import APP_SAMPLE_RATE, PROJECT_SOUND_SEGMENTS_SUBDIR
from tts_audiobook_tool.constants_config import PROJECT_MAX_RETRIES_DEFAULT
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.web_console import capture_synthesis_console, relay_synthesis_console
from tts_audiobook_tool.web_models import ModelRunError, ROOT, _RENDER_LOCK, _python_for, _stop, _write_json
from tts_audiobook_tool.web_remote_models import (
    REMOTE_MODEL_ID, RemoteTtsSelection, WebRemoteTtsError, remote_client_python, remote_worker_scope,
)

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types import Book
    from tts_audiobook_tool.web_backend import BookSession
    from tts_audiobook_tool.web_render import RenderVoice
    from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettings


class WebPlaybackError(ValueError):
    """A playback failure suitable for display in the browser."""


@dataclass(frozen=True)
class PlaybackChunk:
    line: int
    wave: bytes


def playback_range(text: str, count: int) -> tuple[int, int]:
    if not count:
        raise WebPlaybackError("Open a book before starting playback.")
    try:
        parsed = RangeStringUtil.parse_range_string_normal(text or "all", count)
    except ValueError:
        parsed = "Invalid range"
    if isinstance(parsed, str):
        raise WebPlaybackError(f"Choose line numbers from 1 to {count:,}.")
    return (1, count) if parsed == (0, 0) else parsed


def play_book(
        model_id: str, session: BookSession, voices: dict[str, RenderVoice], *, first: int, last: int,
        validate_audio: bool = True, batch_size: int = 1, model_options: dict[str, object] | None = None,
        strictness: str | None = None, max_retries: int = PROJECT_MAX_RETRIES_DEFAULT,
        cancel_event: Event | None = None, runtime_settings: WebRuntimeSettings | None = None,
        remote_selection: RemoteTtsSelection | None = None,
        on_console: Callable[[str], None] | None = None,
        performance_cues: bool = False,
        character_voices: bool = False,
) -> Iterator[PlaybackChunk]:
    """Yield ordered WAV chunks while the selected environment prepares the range."""
    from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict

    cancel = cancel_event or Event()
    if type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(session.book.phrase_groups):
        raise WebPlaybackError("Choose a valid playback range.")
    if type(performance_cues) is not bool:
        raise WebPlaybackError("Choose a valid performance cue setting.")
    if type(character_voices) is not bool:
        raise WebPlaybackError("Choose a valid character voice setting.")
    if (model_id == REMOTE_MODEL_ID) != (remote_selection is not None):
        raise WebPlaybackError("Choose a verified remote model before starting remote playback.")
    while not _RENDER_LOCK.acquire(timeout=0.1):
        _check_cancelled(cancel)
    try:
        _check_cancelled(cancel)
        with (
            TemporaryDirectory(prefix="audiobook-playback-") as temp,
            capture_synthesis_console(Path(temp), on_console) as (console_options, publish_console),
        ):
            directory = Path(temp)
            _write_json(directory / "input.json", {
                "book": book_to_project_text_json_dict(session.book), "project_dir": session.project_dir,
                "roster": session.roster.to_dict() if session.roster is not None else None,
                "attributions": session.attributions.to_dict(), "language": session.project_language,
                "word_substitutions": session.word_substitutions,
                "line_voices": {slot: asdict(voice) for slot, voice in session.line_voices.items()},
                "voices": {role: asdict(voice) for role, voice in voices.items()},
                "first": first, "last": last, "validate_audio": validate_audio, "batch_size": batch_size,
                "model_options": model_options, "strictness": strictness, "max_retries": max_retries,
                "performance_cues": performance_cues,
                "character_voices": character_voices,
                "runtime_settings": asdict(runtime_settings) if runtime_settings is not None else None,
                "remote_selection": asdict(remote_selection) if remote_selection is not None else None,
            })
            try:
                executable = remote_client_python(ROOT) if remote_selection is not None else _python_for(model_id)
                if executable is None:
                    raise WebPlaybackError("No installed remote speech client environment was found.")
                process = subprocess.Popen(
                    [str(executable), "-m", "tts_audiobook_tool.web_playback", temp],
                    cwd=ROOT, stdin=subprocess.DEVNULL, start_new_session=os.name == "posix",
                    **console_options,
                )
            except (OSError, ModelRunError) as error:
                raise WebPlaybackError("Could not start the selected speech environment.") from error
            try:
                for line in range(first, last + 1):
                    wave = _ready_path(directory, line)
                    while not wave.is_file():
                        _check_cancelled(cancel)
                        publish_console()
                        if process.poll() is not None:
                            _verify_worker_result(directory)
                            if not wave.is_file():
                                raise WebPlaybackError(f"Audio is unavailable for line {line:,}.")
                        else:
                            cancel.wait(0.1)
                    _check_cancelled(cancel)
                    publish_console()
                    packet = wave.read_bytes()
                    wave.unlink()
                    yield PlaybackChunk(line, packet)
                while process.poll() is None:
                    _check_cancelled(cancel)
                    publish_console()
                    cancel.wait(0.1)
                _verify_worker_result(directory)
            finally:
                if process.poll() is None:
                    _stop(process)
    finally:
        _RENDER_LOCK.release()


def _ready_path(directory: Path, line: int) -> Path:
    return directory / f"{line:08d}.wav"


def _verify_worker_result(directory: Path) -> None:
    try:
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise WebPlaybackError("The speech environment stopped before the audio could be prepared.") from None
    if result.get("error"):
        raise WebPlaybackError(result["error"])


def _check_cancelled(cancel: Event) -> None:
    if cancel.is_set():
        raise WebPlaybackError("Playback stopped.")


def _best_takes(work: Path, book: Book, first: int, last: int) -> dict[int, Path]:
    from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil, get_segment_error_rank
    from tts_audiobook_tool.sound.checkpoint_audio import has_complete_audio

    audio = work / PROJECT_SOUND_SEGMENTS_SUBDIR
    if work.is_symlink() or audio.is_symlink() or not audio.is_dir():
        return {}
    source = SimpleNamespace(dir_path=str(work), sound_segments_path=str(audio), phrase_groups=book.phrase_groups)
    takes = SoundSegmentUtil.make_sound_segments_map(source)
    result = {}
    for index in range(first - 1, last):
        candidates = sorted(takes.get(index, ()), key=lambda take: get_segment_error_rank(take, audio))
        for take in candidates:
            path = audio / take.file_name
            if not path.is_symlink() and has_complete_audio(path):
                result[index + 1] = path
                break
    return result


class _PlaybackAudio:
    def __init__(self):
        from tts_audiobook_tool.sound.voice_spatial import VoiceSpatialProcessor

        self.spatial = VoiceSpatialProcessor(channels=2)

    def wave(self, path: Path, voice: RenderVoice, pause: float, final_line: bool) -> bytes:
        import numpy as np
        import soundfile

        from tts_audiobook_tool.app_types import RoomReverb, Sound
        from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
        from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline

        take = SoundFileUtil.load(str(path))
        if isinstance(take, str) or not take.data.size:
            raise WebPlaybackError("A generated audio take could not be read.")
        take = SoundPipeline.resample_for_app(take)
        narration = Sound(take.data * (voice.volume / 100), take.sr)
        if pause > 0:
            narration = Sound(np.pad(narration.data, (0, round(pause * take.sr))), take.sr)
        spatial = self.spatial.apply(
            narration, pan=voice.pan, send_end=len(take.data), room_reverb=RoomReverb.get_by_id(voice.room_reverb),
        )
        data = spatial.data
        if final_line and (tail := self.spatial.drain_tail()) is not None:
            data = np.concatenate((data, tail.data))
        output = BytesIO()
        soundfile.write(output, data, take.sr, format="WAV", subtype="PCM_16")
        return output.getvalue()


def wave_to_adts(wave: bytes) -> tuple[bytes, float]:
    """Encode a browser WAV chunk using PyAV's bundled AAC encoder."""
    import av
    import soundfile

    audio, rate = soundfile.read(BytesIO(wave), dtype="float32", always_2d=True)
    output = BytesIO()
    packets_written = 0
    with av.open(output, mode="w", format="adts") as container:
        stream = container.add_stream("aac", rate=rate)
        stream.layout = "stereo"
        stream.bit_rate = 96_000
        for offset in range(0, len(audio), 1024):
            samples = audio[offset:offset + 1024].T.copy()
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.pts = offset
            frame.time_base = Fraction(1, rate)
            packets = stream.encode(frame)
            container.mux(packets)
            packets_written += len(packets)
        packets = stream.encode(None)
        container.mux(packets)
        packets_written += len(packets)
        duration = packets_written * stream.codec_context.frame_size / rate
    return output.getvalue(), duration


def _run_playback(
    directory: Path, session: BookSession, voices: dict[str, RenderVoice], payload: dict, cancel: Event,
    *, remote_selection: RemoteTtsSelection | None = None,
    on_console: Callable[[str], None] | None = None,
) -> None:
    from tts_audiobook_tool.app_types import Strictness
    from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
    from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
    from tts_audiobook_tool.web_render import (
        _cast_book, _checkpoint_id, _detect_model, _render_job, _validated_model_options, render_book,
    )
    from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettings

    first, last = payload["first"], payload["last"]
    validate = payload["validate_audio"]
    strictness = payload["strictness"]
    retries = payload["max_retries"]
    performance_cues = payload.get("performance_cues", False)
    character_voices = payload.get("character_voices", False)
    renderer_options = (
        {"runtime_settings": WebRuntimeSettings(**payload["runtime_settings"])} if payload["runtime_settings"] else {}
    )
    if remote_selection is not None:
        renderer_options["remote_selection"] = remote_selection
    with _render_job(cancel):
        model = TtsModelType.require_by_id(remote_selection.catalog_id) if remote_selection is not None else _detect_model()
        local_breeze = remote_selection is None and model.id == BreezeBaseModel.INFO.id
        book, references = _cast_book(
            session, voices,
            performance_cues=performance_cues and local_breeze,
            character_voices=character_voices and local_breeze,
        )
        options = _validated_model_options(model.id, payload["model_options"])
        batch = payload["batch_size"] if model.can_batch() else 1
        run_id = _checkpoint_id(
            book, references, model.id, session.language_code, validate, batch, model_options=options,
            word_substitutions=session.word_substitutions, strictness=Strictness.get_by_id(strictness),
            max_retries=retries, **renderer_options,
        )
    root = Path(session.project_dir) / "renders"
    existing = _best_takes(root / run_id, book, first, last) if not root.is_symlink() else {}
    encoder = _PlaybackAudio()
    pauses = ReasonPauseTypes.default().value
    next_line = first

    def send_ready(takes: dict[int, Path]) -> None:
        nonlocal next_line
        while next_line <= last and next_line in takes:
            _check_cancelled(cancel)
            # Limit staged browser chunks without blocking the model's disk writes.
            while next_line >= first + 12 and _ready_path(directory, next_line - 12).exists():
                _check_cancelled(cancel)
                cancel.wait(0.1)
            group = book.phrase_groups[next_line - 1]
            pause = pauses.get_pause_for(group.phrases[-1].reason) if next_line < last and group.phrases else 0.0
            wave = encoder.wave(takes[next_line], references[group.voice_index], pause, next_line == last)
            path = _ready_path(directory, next_line)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(wave)
            temporary.replace(path)
            next_line += 1

    send_ready(existing)
    if next_line > last:
        return
    # Keep new synthesis temporary. Reviewable book takes and their manifests
    # stay independent of a playback session.
    work = directory / "render"
    stage = work / "renders" / run_id
    audio = stage / PROJECT_SOUND_SEGMENTS_SUBDIR
    audio.mkdir(parents=True)
    for line, path in existing.items():
        if line >= next_line:
            _check_cancelled(cancel)
            shutil.copyfile(path, audio / path.name)

    def report(_current: int, _total: int) -> None:
        _check_cancelled(cancel)
        if next_line <= last:
            send_ready(_best_takes(stage, book, next_line, last))

    render_book(
        replace(session, project_dir=str(work)), voices, validate_audio=validate, batch_size=batch,
        model_options=options, strictness=strictness, max_retries=retries, first_line=next_line, last_line=last,
        performance_cues=performance_cues, character_voices=character_voices,
        auto_concat=False, cancel_event=cancel, on_progress=report, on_console=on_console, **renderer_options,
    )
    report(0, 0)
    if next_line <= last:
        raise WebPlaybackError(f"Audio is unavailable for line {next_line:,}.")


def _worker(directory: Path) -> None:
    from tts_audiobook_tool import app_support  # Initialize shared imports first.
    from tts_audiobook_tool.app_types.book_serialization import book_from_project_text_json_dict
    from tts_audiobook_tool.model_worker import ModelWorker
    from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
    from tts_audiobook_tool.speaker_assign.roster import Roster
    from tts_audiobook_tool.web_backend import BookSession
    from tts_audiobook_tool.web_render import RenderVoice, WebRenderError
    from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettingsError

    cancel = Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    try:
        payload = json.loads((directory / "input.json").read_text(encoding="utf-8"))
        book = book_from_project_text_json_dict(payload["book"])
        if isinstance(book, str):
            raise WebPlaybackError("Could not read the book selected for playback.")
        session = BookSession(
            book, payload["project_dir"], roster=Roster.from_dict(payload["roster"]) if payload["roster"] else None,
            attributions=AttributionsFile.from_dict(payload["attributions"]), project_language=payload["language"],
            word_substitutions=payload["word_substitutions"],
            line_voices={int(slot): RenderVoice(**voice) for slot, voice in payload["line_voices"].items()},
        )
        voices = {role: RenderVoice(**voice) for role, voice in payload["voices"].items()}
        data = payload.get("remote_selection")
        remote = RemoteTtsSelection(**data) if data is not None else None
        with remote_worker_scope(remote):
            _run_playback(
                directory, session, voices, payload, cancel, remote_selection=remote,
                on_console=relay_synthesis_console(directory),
            )
    except (WebPlaybackError, WebRenderError, WebRuntimeSettingsError, WebRemoteTtsError) as error:
        _write_json(directory / "result.json", {"error": str(error)})
    except Exception:
        _write_json(directory / "result.json", {"error": "Could not prepare playback audio."})
    else:
        _write_json(directory / "result.json", {})
    finally:
        ModelWorker.shutdown()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
