"""Use the existing model worker to render audio for the local web interface."""
from __future__ import annotations

import atexit
import hashlib
import json
import multiprocessing
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from os import SEEK_END
from pathlib import Path
from typing import TYPE_CHECKING

from tts_audiobook_tool import readiness
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Book, ExportType, Sound, SttConfig, SttVariant
from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.constants import PROJECT_JSON_FILE_NAME
from tts_audiobook_tool.constants_config import (
    PROJECT_BATCH_SIZE_DEFAULT,
    PROJECT_BATCH_SIZE_MAX,
    VOICE_CLONE_TRANSCRIBE_MIN_PROBABILITY,
)
from tts_audiobook_tool.generation_events import GenerationProgress, GenerationTimedOut, ModelUnhealthy
from tts_audiobook_tool.l import L
from tts_audiobook_tool.model_runtime import mark_interactive_main
from tts_audiobook_tool.model_worker import ModelWorker, ModelWorkerUnavailable
from tts_audiobook_tool.model_worker_protocol import (
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    TtsPreviewFinished,
    WorkerCommandCancelled,
    WorkerCommandFailed,
    WorkerExited,
)
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

if TYPE_CHECKING:
    from tts_audiobook_tool.web_backend import BookSession


@dataclass(frozen=True)
class RenderVoice:
    reference_path: str | None = None
    transcript: str = ""
    instructions: str = ""


class WebRenderError(ValueError):
    """An error that can be shown in a web callback."""


_render_lock = threading.Lock()
_CANCEL_GRACE_SECONDS = 2.0
_NARRATOR = "NARRATOR"

# Initialize multiprocessing's exit hook first so it cannot join the
# non-daemon model worker before our shutdown hook has stopped it.
multiprocessing.get_logger()
atexit.register(ModelWorker.shutdown)


@cache
def _detect_model() -> TtsModelType:
    model, count = Tts.init_local_model_type()
    if count != 1 or model.id == "none":
        raise WebRenderError("Start the interface in an environment with one local TTS model.")
    return model


def get_local_model_label() -> str:
    try:
        model = _detect_model()
    except WebRenderError:
        return "No local TTS model"
    return model.value.ui.get("proper_name") or model.id


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise WebRenderError("Rendering stopped.")


@contextmanager
def _render_job(cancel_event: threading.Event | None) -> Iterator[None]:
    while not _render_lock.acquire(timeout=0.1):
        _check_cancelled(cancel_event)
    try:
        _check_cancelled(cancel_event)
        mark_interactive_main()
        if not hasattr(L, "logger"):
            L.init("tts-audiobook-web")
        yield
    finally:
        _render_lock.release()


def _build_voice(project: Project, index: int, settings: RenderVoice) -> Voice:
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline

    support = Tts.get_model_support(project)
    instructions = settings.instructions.strip()
    if instructions and support.can_instructions(project):
        if message := support.validate_instructions(instructions):
            raise WebRenderError(message)
        instructions_by_model = {project.tts_model_type: instructions}
    else:
        instructions_by_model = {}
    file_name = ""
    if settings.reference_path and support.can_voice_clone(project):
        source = Path(settings.reference_path)
        if not source.is_file():
            raise WebRenderError(f"Voice {index + 1} reference audio is unavailable.")
        sound = SoundFileUtil.load(str(source))
        if isinstance(sound, str):
            L.w(sound)
            raise WebRenderError(f"Voice {index + 1} reference audio could not be read.")
        sound = SoundPipeline.apply_voice_clone_post_processing(sound)
        if not sound.data.size:
            raise WebRenderError(f"Voice {index + 1} reference audio contains no audible sound.")
        voice_dir = Path(ProjectVoiceUtil.get_voice_dir_path(project))
        voice_dir.mkdir(exist_ok=True)
        file_name = f"voice-{index + 1}.flac"
        error = SoundFileUtil.save_flac(sound, str(voice_dir / file_name))
        if error:
            L.w(error)
            raise WebRenderError(f"Voice {index + 1} reference audio could not be prepared.")
    return Voice(
        file_name=file_name,
        transcript=settings.transcript.strip() if file_name else "",
        instructions=instructions_by_model,
    )


@contextmanager
def _project_state(
    project_dir: Path,
    language_code: str,
    voices: list[RenderVoice],
    book: Book | None = None,
    *,
    validate_audio: bool = False,
) -> Iterator[State]:
    from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

    model = _detect_model()
    project = None
    try:
        if (project_dir / PROJECT_JSON_FILE_NAME).is_file():
            loaded = ProjectLoadUtil.load_using_dir_path(str(project_dir), prompt_on_warnings=False)
            if isinstance(loaded, str):
                L.w(loaded)
                raise WebRenderError("The saved rendering project could not be opened.")
            project = loaded
        else:
            project = Project(
                dir_path=str(project_dir),
                language_code=language_code,
                tts_model_type=model.id,
                export_type=ExportType.AAC,
                gen_auto_concat=False,
            )
            if book is not None:
                project.book = book
            project.voice_references = [
                _build_voice(project, index, voice) for index, voice in enumerate(voices)
            ]
            if book is not None and (error := ProjectTextIOUtil.save_book(project)):
                L.w(error)
                raise WebRenderError("The audiobook text could not be prepared.")
            if error := project.save():
                L.w(error)
                raise WebRenderError("The rendering project could not be prepared.")

        stt_variant = SttVariant.get_default() if validate_audio else SttVariant.DISABLED
        state = State.for_worker(Prefs(project_dir=str(project_dir), stt_variant=stt_variant))
        state.project = project
        blockers = readiness.get_tts_blockers(project)
        if blockers:
            raise WebRenderError(blockers[0].verbose)
        indices = {group.voice_index or 0 for group in book.phrase_groups} if book else {0}
        if problems := VoiceMenuShared.get_voice_problems(state, voice_indices=indices):
            raise WebRenderError(problems[0])
        yield state
    finally:
        if project is not None:
            project.kill()
            project.sound_segments.observer.join()


def _wait_for_job(
    operation_id: str,
    result_type: type[GenerationFinished] | type[TtsPreviewFinished],
    cancel_event: threading.Event | None,
    on_progress: Callable[[int, int], None] | None = None,
) -> GenerationFinished | TtsPreviewFinished:
    cancellation_deadline = None
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                if cancellation_deadline is None:
                    ModelWorker.request_cancel(operation_id)
                    cancellation_deadline = time.monotonic() + _CANCEL_GRACE_SECONDS
                elif time.monotonic() >= cancellation_deadline:
                    raise WebRenderError("Rendering stopped.")

            event = ModelWorker.get_event(timeout=0.1)
            if getattr(event, "operation_id", None) != operation_id:
                continue
            if isinstance(event, result_type):
                _check_cancelled(cancel_event)
                if event.status is GenerationTerminalStatus.COMPLETED:
                    return event
                if event.message:
                    L.w(event.message)
                if event.status is GenerationTerminalStatus.CANCELLED:
                    raise WebRenderError("Rendering stopped.")
                raise WebRenderError("The model could not finish rendering the audio.")
            if isinstance(event, WorkerCommandCancelled):
                raise WebRenderError("Rendering stopped.")
            if isinstance(event, (WorkerExited, WorkerCommandFailed)):
                L.w(event.message)
                raise WebRenderError("The model worker stopped before rendering completed.")
            if isinstance(event, GenerationUpdate):
                update = event.update
                if isinstance(update, GenerationTimedOut):
                    raise WebRenderError("The model took too long to generate an audio segment.")
                if isinstance(update, ModelUnhealthy):
                    L.w(update.reason)
                    raise WebRenderError("The model could not continue rendering.")
                if on_progress is not None and isinstance(update, GenerationProgress):
                    on_progress(update.processed, update.total)
    finally:
        if ModelWorker.is_busy():
            # End a hung or cancelled job before removing its temporary project.
            ModelWorker.shutdown()


def render_sample(
    text: str,
    voice: RenderVoice,
    *,
    language_code: str = "en",
    cancel_event: threading.Event | None = None,
) -> str:
    """Render a single prompt with the active environment's model to a WAV file."""
    import soundfile

    text = text.strip()
    if not text:
        raise WebRenderError("Enter some text to preview.")
    with (
        _render_job(cancel_event),
        tempfile.TemporaryDirectory(prefix="tts-web-preview-") as work,
        _project_state(Path(work), language_code, [voice]) as state,
    ):
        _check_cancelled(cancel_event)
        try:
            operation_id = ModelWorker.submit_tts_preview(state=state, prompt=text)
        except ModelWorkerUnavailable as error:
            L.w(str(error))
            raise WebRenderError("The model worker could not start.") from error
        event = _wait_for_job(operation_id, TtsPreviewFinished, cancel_event)
        if not isinstance(event, TtsPreviewFinished) or not isinstance(event.sound, Sound):
            raise WebRenderError("The model returned no playable audio.")
        with tempfile.NamedTemporaryFile(prefix="tts-preview-", suffix=".wav", delete=False) as output:
            output_path = Path(output.name)
        try:
            soundfile.write(str(output_path), event.sound.data, event.sound.sr, subtype="PCM_16")
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
        return str(output_path)



def transcribe_reference(path: str, *, language_code: str = "en") -> str:
    """Transcribe a reference clip using the local CPU Whisper cache."""
    from faster_whisper.utils import download_model
    from huggingface_hub.errors import LocalEntryNotFoundError

    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
    from tts_audiobook_tool.stt import FasterWhisperAdapter
    from tts_audiobook_tool.text_ops.language_util import normalize_language_code
    from tts_audiobook_tool.transcriber import Transcriber

    with _render_job(None):
        if not path or not Path(path).is_file():
            raise WebRenderError("Upload a reference audio clip first.")
        sound = SoundFileUtil.load(path)
        if isinstance(sound, str):
            L.w(sound)
            raise WebRenderError("The reference audio could not be read.")
        sound = SoundPipeline.apply_voice_clone_post_processing(sound)
        if not sound.data.size:
            raise WebRenderError("The reference audio contains no audible sound.")
        sound = Transcriber.prepare_sound_for_whisper(sound)

        try:
            snapshot = Path(download_model(SttVariant.get_default().id, local_files_only=True))
        except LocalEntryNotFoundError as error:
            raise WebRenderError("The local Whisper model is unavailable. Enter the transcript manually.") from error
        # Faster Whisper falls back to fetching a tokenizer if the local file is missing.
        if not all((snapshot / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")):
            raise WebRenderError("The local Whisper model is incomplete. Enter the transcript manually.")

        config = SttConfig.CPU_INT8FLOAT32
        try:
            whisper = FasterWhisperAdapter(
                str(snapshot), device=config.device, compute_type=config.compute_type, cpu_threads=4,
            )
            language = normalize_language_code(language_code)
            segments, _ = whisper.transcribe(
                sound.data,
                language=language if language in whisper.supported_languages else None,
                word_timestamps=True,
            )
            words = Transcriber.get_words_from_segments(segments)
        except Exception as error:
            L.w(str(error))
            raise WebRenderError("Whisper could not transcribe the reference audio.") from error
        transcript = Transcriber.get_flat_text_filtered_by_probability(words, VOICE_CLONE_TRANSCRIBE_MIN_PROBABILITY)
        if not transcript:
            raise WebRenderError("Whisper found no clear words in the reference audio.")
        return transcript


def _cast_book(session: BookSession, voices: dict[str, RenderVoice]) -> tuple[Book, list[RenderVoice]]:
    narrator = session.roster.narrator if session.roster else None
    narrator_id = narrator.id if narrator else _NARRATOR
    narrator_voice = voices.get(narrator_id, voices.get(_NARRATOR, RenderVoice()))
    speakers = session.attributions.speaker_of_each_line(session.book)
    book = deepcopy(session.book)
    references = [narrator_voice]
    voice_indices = {narrator_voice: 0}
    for index, group in enumerate(book.phrase_groups):
        speaker = speakers.get(index, narrator_id)
        settings = voices.get(speaker, narrator_voice)
        if settings not in voice_indices:
            voice_indices[settings] = len(references)
            references.append(settings)
        group.voice_index = voice_indices[settings]
    return book, references


def _checkpoint_id(
    book: Book,
    references: list[RenderVoice],
    model_id: str,
    language_code: str,
    validate_audio: bool,
    batch_size: int,
) -> str:
    voice_settings = []
    for index, voice in enumerate(references):
        clip_digest = ""
        if voice.reference_path:
            try:
                with Path(voice.reference_path).open("rb") as clip:
                    clip_digest = hashlib.file_digest(clip, "sha256").hexdigest()
            except OSError as error:
                raise WebRenderError(f"Voice {index + 1} reference audio is unavailable.") from error
        voice_settings.append((clip_digest, voice.transcript.strip(), voice.instructions.strip()))

    payload = {
        "version": 1,
        "book": book_to_project_text_json_dict(book),
        "voices": voice_settings,
        "model": model_id,
        "catalog": Tts._config_fingerprint,
        "language": language_code,
        "validate_audio": validate_audio,
        "batch_size": batch_size,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _missing_segments(project: Project) -> set[int]:
    """Scan saved takes, discarding files left incomplete by an interrupted write."""
    import soundfile

    project.sound_segments.force_invalidate()
    missing = set()
    for index in range(len(project.phrase_groups)):
        while file_name := project.sound_segments.get_best_file_for(index):
            path = Path(project.sound_segments_path) / file_name
            try:
                with soundfile.SoundFile(path) as audio:
                    audio.seek(-1, SEEK_END)
                    if audio.read(1).size:
                        break
            except (OSError, ValueError, soundfile.LibsndfileError):
                pass
            L.w(f"Discarding unreadable segment {file_name} for resumed generation")
            path.unlink(missing_ok=True)
            project.sound_segments.force_invalidate()
        else:
            missing.add(index)
    return missing


def render_book(
    session: BookSession,
    voices: dict[str, RenderVoice],
    *,
    language_code: str | None = None,
    validate_audio: bool = True,
    batch_size: int = PROJECT_BATCH_SIZE_DEFAULT,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Resume matching saved segments, then export a complete M4B audiobook."""
    from filelock import FileLock, Timeout

    if not session.book.phrase_groups:
        raise WebRenderError("Import an EPUB with readable text first.")
    if not 1 <= batch_size <= PROJECT_BATCH_SIZE_MAX:
        raise WebRenderError(f"Choose a batch size from 1 to {PROJECT_BATCH_SIZE_MAX}.")
    with _render_job(cancel_event):
        model = _detect_model()
        if not model.can_batch():
            batch_size = 1
        book, references = _cast_book(session, voices)
        language_code = language_code or book.segmentation_settings.language_code or "en"
        checkpoint = _checkpoint_id(
            book, references, model.id, language_code, validate_audio, batch_size,
        )
        work = Path(session.project_dir) / "renders" / checkpoint
        work.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(work / ".render.lock"), timeout=0)
        try:
            with lock:
                return _render_book_checkpoint(
                    work, book, references, language_code, validate_audio,
                    batch_size, on_progress, cancel_event,
                )
        except Timeout as error:
            raise WebRenderError("This audiobook is already rendering.") from error


def _render_book_checkpoint(
    work: Path,
    book: Book,
    references: list[RenderVoice],
    language_code: str,
    validate_audio: bool,
    batch_size: int,
    on_progress: Callable[[int, int], None] | None,
    cancel_event: threading.Event | None,
) -> str:
    from tts_audiobook_tool.concat_util import ConcatUtil

    _check_cancelled(cancel_event)
    total = len(book.phrase_groups)
    output = work / "audiobook.abr.m4b"
    if output.is_file():
        if on_progress is not None:
            on_progress(total, total)
        return str(output)

    with _project_state(work, language_code, references, book, validate_audio=validate_audio) as state:
        remaining = _missing_segments(state.project)
        if on_progress is not None:
            on_progress(total - len(remaining), total)
        _check_cancelled(cancel_event)
        if remaining:
            def report_progress(processed: int, requested: int) -> None:
                if on_progress is not None:
                    on_progress(total - requested + processed, total)

            try:
                operation_id = ModelWorker.submit_generation(
                    state=state, indices=remaining, batch_size=batch_size, is_regen=False,
                )
            except ModelWorkerUnavailable as error:
                L.w(str(error))
                raise WebRenderError("The model worker could not start.") from error
            _wait_for_job(operation_id, GenerationFinished, cancel_event, report_progress)
            if missing := _missing_segments(state.project):
                raise WebRenderError(
                    f"The model did not produce audio for {len(missing)} of {total} text segments."
                )

        _check_cancelled(cancel_event)
        staging_dir = work / "export"
        staging_dir.mkdir(exist_ok=True)
        interrupts = Interrupts()
        interrupts.set_external_event(cancel_event)
        try:
            path, error = ConcatUtil.make_file(
                state=state,
                index_start=0,
                index_end=total - 1,
                bookmark_indices=[],
                stem_path=str(staging_dir / "audiobook"),
                use_upsampler=False,
            )
        finally:
            interrupts.set_external_event(None)
        _check_cancelled(cancel_event)
        if error:
            L.w(error)
            raise WebRenderError("The audiobook audio could not be exported.")
        Path(path).replace(output)
        return str(output)
