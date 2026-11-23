"""Use the existing model worker to render audio for the local web interface."""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import multiprocessing
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cache
from os import SEEK_END
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from tts_audiobook_tool import readiness
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Book, ExportType, RoomReverb, Sound, Strictness, SttConfig, SttVariant
from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict
from tts_audiobook_tool.app_types.voice import VOLUME_DEFAULT, VOLUME_MAX, VOLUME_MIN, Voice
from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, PROJECT_JSON_FILE_NAME
from tts_audiobook_tool.constants_config import (
    IMPORT_LIMIT_SILENCE_DURATION_MAX,
    IMPORT_LIMIT_SILENCE_DURATION_MIN,
    PROJECT_BATCH_SIZE_DEFAULT,
    PROJECT_BATCH_SIZE_MAX,
    PROJECT_MAX_RETRIES_DEFAULT,
    PROJECT_MAX_RETRIES_MAX,
    PROJECT_MAX_RETRIES_MIN,
    VOICE_CLONE_TRANSCRIBE_MIN_PROBABILITY,
)
from tts_audiobook_tool.generation_events import (
    BookRenderTelemetry,
    GenerationPhase,
    GenerationProgress,
    GenerationStats,
    GenerationTimedOut,
    ModelUnhealthy,
)
from tts_audiobook_tool.l import L
from tts_audiobook_tool.model_runtime import mark_interactive_main
from tts_audiobook_tool.model_worker import ModelWorker, ModelWorkerUnavailable
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleOutput,
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    TtsPreviewFinished,
    WorkerCommandCancelled,
    WorkerCommandFailed,
    WorkerExited,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_backend import UNIDENTIFIED_CAST_ROLES, canonical_cast_role
from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettings, WebRuntimeSettingsError

if TYPE_CHECKING:
    from tts_audiobook_tool.textual.generation_app import GenerationTranscript
    from tts_audiobook_tool.web_backend import BookSession
    from tts_audiobook_tool.web_remote_models import RemoteTtsSelection


class WebRenderError(ValueError):
    """An error that can be shown in a web callback."""


def _validate_trim(trim_start: float, trim_end: float | None) -> None:
    if type(trim_start) not in (int, float) or not 0 <= trim_start < math.inf:
        raise WebRenderError("Trim start must be a nonnegative number of seconds.")
    if trim_end is not None and (type(trim_end) not in (int, float) or not trim_start < trim_end < math.inf):
        raise WebRenderError("Trim end must be a number of seconds greater than the start.")


def _model_pairs(value: object, error: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (tuple, list)):
        raise WebRenderError(error)
    items = {}
    for pair in value:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or not isinstance(pair[0], str) or not pair[0] or not isinstance(pair[1], str)):
            raise WebRenderError(error)
        items[pair[0]] = pair[1]
    return tuple(sorted(items.items()))


@dataclass(frozen=True)
class RenderVoice:
    reference_path: str | None = None
    transcript: str = ""
    instructions: str = ""
    name: str = ""
    trim_start: float = 0.0
    trim_end: float | None = None
    pan: float = 0.0
    volume: int = VOLUME_DEFAULT
    room_reverb: str = RoomReverb.OFF.id
    model_instructions: tuple[tuple[str, str], ...] = ()
    presets: tuple[tuple[str, str], ...] = ()
    max_reference_silence: float | None = None

    def __post_init__(self) -> None:
        if self.reference_path is not None and not isinstance(self.reference_path, str):
            raise WebRenderError("The reference audio path must be text.")
        if not all(isinstance(value, str) for value in (self.name, self.transcript, self.instructions)):
            raise WebRenderError("Voice name, transcript and instructions must be text.")
        _validate_trim(self.trim_start, self.trim_end)
        if type(self.pan) not in (int, float) or not -1 <= self.pan <= 1:
            raise WebRenderError("Stereo position must be between -1 and 1.")
        if type(self.volume) is not int or not VOLUME_MIN <= self.volume <= VOLUME_MAX:
            raise WebRenderError(f"Voice volume must be a whole number from {VOLUME_MIN} to {VOLUME_MAX}.")
        if not isinstance(self.room_reverb, str) or RoomReverb.get_by_id(self.room_reverb) is None:
            raise WebRenderError("Choose a room reverb preset.")
        instructions = _model_pairs(self.model_instructions, "Each set of instructions needs a speech model and text.")
        presets = _model_pairs(self.presets, "Each preset voice needs a speech model and a preset name.")
        object.__setattr__(self, "model_instructions", instructions)
        object.__setattr__(self, "presets", tuple((model, preset.strip()) for model, preset in presets if preset.strip()))
        object.__setattr__(self, "trim_start", float(self.trim_start))
        if self.trim_end is not None:
            object.__setattr__(self, "trim_end", float(self.trim_end))
        object.__setattr__(self, "pan", float(self.pan))
        gap = self.max_reference_silence
        if gap is not None:
            if type(gap) not in (float, int) or not IMPORT_LIMIT_SILENCE_DURATION_MIN <= gap <= IMPORT_LIMIT_SILENCE_DURATION_MAX:
                raise WebRenderError(
                    f"Choose a reference silence limit of {IMPORT_LIMIT_SILENCE_DURATION_MIN:g}"
                    f" to {IMPORT_LIMIT_SILENCE_DURATION_MAX:g} seconds."
                )
            object.__setattr__(self, "max_reference_silence", float(gap))


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


def _validated_model_options(model_id: str, options: dict[str, object] | None) -> dict[str, object]:
    if model_id != "pocket_local" and (options is None or options == {}):
        return {}

    from tts_audiobook_tool.web_model_options import POCKET_DEFAULT_MODEL_CODE, validate_model_options

    try:
        validated = validate_model_options(model_id, options)
    except ValueError as error:
        raise WebRenderError(str(error)) from error
    if model_id == "pocket_local":
        # The native project requires a concrete code, including for the browser default.
        validated.setdefault("model_code", POCKET_DEFAULT_MODEL_CODE)
    return validated


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


def _voice_label(index: int, name: str) -> str:
    name = " ".join(name.split())
    return f'Voice "{name}"' if name else f"Voice {index + 1}"


def _build_voice(project: Project, index: int, settings: RenderVoice, *, label: str | None = None) -> Voice:
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline

    label = label or _voice_label(index, settings.name)
    support = Tts.get_model_support(project)
    model_id = project.tts_model_type
    presets = dict(settings.presets)
    active_preset = presets.get(model_id, "") if support.can_voice_presets(project) else ""
    if active_preset:
        from tts_audiobook_tool.web_model_options import voice_presets_for

        known_presets = voice_presets_for(model_id)
        if known_presets:
            normalized = active_preset.casefold()
            if normalized not in known_presets:
                raise WebRenderError(f"{label} needs a built-in voice from the selected model.")
            presets[model_id] = normalized
    instructions_by_model = {model: text.strip() for model, text in settings.model_instructions}
    if support.can_instructions(project):
        instructions = instructions_by_model.get(project.tts_model_type, settings.instructions.strip())
        if instructions:
            if message := support.validate_instructions(instructions):
                raise WebRenderError(message)
            instructions_by_model[project.tts_model_type] = instructions
    file_name = ""
    if settings.reference_path and not active_preset and support.can_voice_clone(project):
        sound = _load_reference_audio(
            settings.reference_path, settings.trim_start, settings.trim_end,
            label=f"{label} reference audio",
        )
        sound = SoundPipeline.apply_voice_clone_post_processing(sound, max_silence_gap=settings.max_reference_silence)
        if not sound.data.size:
            raise WebRenderError(f"{label} reference audio contains no audible sound.")
        voice_dir = Path(ProjectVoiceUtil.get_voice_dir_path(project))
        voice_dir.mkdir(exist_ok=True)
        file_name = f"voice-{index + 1}.flac"
        error = SoundFileUtil.save_flac(sound, str(voice_dir / file_name))
        if error:
            L.w(error)
            raise WebRenderError(f"{label} reference audio could not be prepared.")
    return Voice(
        file_name=file_name,
        transcript=settings.transcript.strip() if file_name else "",
        instructions=instructions_by_model,
        presets=presets,
        name=settings.name,
        pan=settings.pan,
        volume=settings.volume,
        room_reverb=RoomReverb.get_by_id(settings.room_reverb) or RoomReverb.OFF,
    )


@contextmanager
def _project_state(
    project_dir: Path,
    language_code: str,
    voices: list[RenderVoice],
    book: Book | None = None,
    *,
    model: TtsModelType,
    model_options: dict[str, object],
    validate_audio: bool = False,
    word_substitutions: dict[str, str] | None = None,
    generation_indices: Collection[int] | None = None,
    strictness: Strictness | None = None,
    max_retries: int = PROJECT_MAX_RETRIES_DEFAULT,
    runtime_settings: WebRuntimeSettings = WebRuntimeSettings(),
    remote_selection: RemoteTtsSelection | None = None,
    voice_labels: list[str] | None = None,
    on_console: Callable[[str], None] | None = None,
) -> Iterator[State]:
    from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

    labels = voice_labels or [_voice_label(index, voice.name) for index, voice in enumerate(voices)]
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
                word_substitutions=word_substitutions or {},
                strictness=strictness.id if strictness is not None else "",
                max_retries=max_retries,
            )
            if remote_selection is not None and remote_selection.backend_kind is TtsBackendKind.AUDIO_CPP:
                project.set_tts_selection(model.id, remote_selection.server_model_id)
            for key, value in model_options.items():
                project.set_model_setting(model.id, key, value)
            if book is not None:
                project.book = book
            project.voice_references = [
                _build_voice(project, index, voice, label=labels[index]) for index, voice in enumerate(voices)
            ]
            if book is not None and (error := ProjectTextIOUtil.save_book(project)):
                L.w(error)
                raise WebRenderError("The audiobook text could not be prepared.")
            if error := project.save():
                L.w(error)
                raise WebRenderError("The rendering project could not be prepared.")

        prefs = runtime_settings.generation_prefs(str(project_dir), validate_audio=validate_audio)
        if remote_selection is not None:
            prefs.remote_tts_url = remote_selection.endpoint
            prefs.tts_force_cpu = False
        state = State.for_worker(prefs)
        if runtime_settings.force_cpu and remote_selection is None:
            try:
                Tts.get_best_supported_device_type(model)
            except RuntimeError as error:
                raise WebRenderError(str(error)) from error
        state.project = project
        blockers = readiness.get_tts_blockers(project)
        if blockers:
            raise WebRenderError(blockers[0].verbose)
        voice_indices = {0}
        if book is not None:
            indices = range(len(book.phrase_groups)) if generation_indices is None else generation_indices
            voice_indices = {book.phrase_groups[index].voice_index or 0 for index in indices}
        if problems := VoiceMenuShared.get_voice_problems(state, voice_indices=voice_indices):
            raise WebRenderError(_describe_voice_problem(project, problems[0], voice_indices, labels))
        if on_console is not None:
            on_console("Voice reference checks passed.\n")
        yield state
    finally:
        if project is not None:
            project.kill()
            project.sound_segments.observer.join()


def _describe_voice_problem(project: Project, problem: str, indices: set[int], labels: list[str]) -> str:
    for index in sorted(indices):
        if not 0 <= index < len(labels):
            continue
        label = labels[index]
        numbered_label = f"Voice {index + 1} "
        if problem.startswith(numbered_label):
            return f"{label} {problem[len(numbered_label):]}"
        if index >= len(project.voice_references):
            continue
        voice = project.voice_references[index]
        file_label = (
            f"Trimmed voice file for {voice.file_name}" if voice.crop is not None else f"Voice file {voice.file_name}"
        )
        if voice.file_name and problem.startswith(f"{file_label} "):
            detail = problem[len(file_label) + 1:]
            if detail == "has no transcript":
                return f"{label} is missing its reference transcript. Add the transcript and save the voice assignment."
            return f"{label} reference audio {detail}."
    return problem


def _wait_for_job(
    operation_id: str,
    result_type: type[GenerationFinished] | type[TtsPreviewFinished],
    cancel_event: threading.Event | None,
    on_progress: Callable[[GenerationProgress], None] | None = None,
    *,
    on_stats: Callable[[GenerationStats], None] | None = None,
    transcript: GenerationTranscript | None = None,
    on_console: Callable[[str], None] | None = None,
) -> GenerationFinished | TtsPreviewFinished:
    cancellation_deadline = None
    reported_phases: set[str] = set()

    def relay_output(event: ConsoleOutput) -> None:
        if transcript is not None:
            transcript.write_chunk(event.text, event.stream)
        if on_console is not None:
            on_console(event.text)

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
            if isinstance(event, ConsoleOutput):
                relay_output(event)
            if isinstance(event, result_type):
                if isinstance(event, GenerationFinished) and (on_console is not None or transcript is not None):
                    from tts_audiobook_tool.textual.worker_session import FINAL_OUTPUT_SETTLE_SECONDS

                    time.sleep(FINAL_OUTPUT_SETTLE_SECONDS)
                    for trailing in ModelWorker.drain_events():
                        if isinstance(trailing, ConsoleOutput) and trailing.operation_id == operation_id:
                            relay_output(trailing)
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
                if on_console is not None and isinstance(update, GenerationPhase) and update.label not in reported_phases:
                    reported_phases.add(update.label)
                    on_console(f"{update.label}\n")
                if on_progress is not None and isinstance(update, GenerationProgress):
                    on_progress(update)
                if on_stats is not None and isinstance(update, GenerationStats):
                    on_stats(update)
    finally:
        if ModelWorker.is_busy():
            # End a hung or cancelled job before removing its temporary project.
            ModelWorker.shutdown()


def _write_preview_wav(sound: Sound, prefix: str) -> str:
    import soundfile

    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False) as output:
        output_path = Path(output.name)
    try:
        soundfile.write(str(output_path), sound.data, sound.sr, subtype="PCM_16")
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return str(output_path)


def _apply_sample_effects(sound: Sound, voice: Voice) -> Sound:
    if voice.pan == 0 and voice.volume == VOLUME_DEFAULT and voice.room_reverb is RoomReverb.OFF:
        return sound

    import numpy as np

    from tts_audiobook_tool.sound.voice_spatial import VoiceSpatialProcessor

    sound = Sound(sound.data * (voice.volume / 100), sound.sr)
    spatial = VoiceSpatialProcessor(channels=2 if voice.pan else 1)
    sound = spatial.apply(sound, pan=voice.pan, room_reverb=voice.room_reverb)
    tail = spatial.drain_tail(dtype=sound.data.dtype)
    return Sound(np.concatenate((sound.data, tail.data)), sound.sr) if tail is not None else sound


def render_sample(
    text: str,
    voice: RenderVoice,
    *,
    language_code: str = "en",
    cancel_event: threading.Event | None = None,
    model_options: dict[str, object] | None = None,
    runtime_settings: WebRuntimeSettings | None = None,
    remote_selection: RemoteTtsSelection | None = None,
    on_console: Callable[[str], None] | None = None,
) -> str:
    """Render a single prompt with the active environment's model to a WAV file."""
    text = text.strip()
    if not text:
        raise WebRenderError("Enter some text to preview.")
    with _render_job(cancel_event):
        model = TtsModelType.require_by_id(remote_selection.catalog_id) if remote_selection else _detect_model()
        options = _validated_model_options(model.id, model_options)
        with (
            tempfile.TemporaryDirectory(prefix="tts-web-preview-") as work,
            _project_state(
                Path(work), language_code, [voice], model=model, model_options=options,
                runtime_settings=runtime_settings or WebRuntimeSettings(),
                remote_selection=remote_selection,
            ) as state,
        ):
            _check_cancelled(cancel_event)
            try:
                operation_id = ModelWorker.submit_tts_preview(state=state, prompt=text)
            except ModelWorkerUnavailable as error:
                L.w(str(error))
                raise WebRenderError("The model worker could not start.") from error
            event = _wait_for_job(operation_id, TtsPreviewFinished, cancel_event, on_console=on_console)
            if not isinstance(event, TtsPreviewFinished) or not isinstance(event.sound, Sound):
                raise WebRenderError("The model returned no playable audio.")
            sound = _apply_sample_effects(event.sound, state.project.voice_references[0])
            return _write_preview_wav(sound, "tts-preview-")


def _load_reference_audio(
    path: str,
    trim_start: float,
    trim_end: float | None,
    *,
    label: str = "The reference audio",
) -> Sound:
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
    from tts_audiobook_tool.sound.sound_util import SoundUtil

    _validate_trim(trim_start, trim_end)
    if not path:
        raise WebRenderError("Upload a reference audio clip first.")
    if not Path(path).is_file():
        raise WebRenderError(f"{label} is unavailable.")
    sound = SoundFileUtil.load(path)
    if isinstance(sound, str):
        if hasattr(L, "logger"):
            L.w(sound)
        raise WebRenderError(f"{label} could not be read.")
    if trim_start == 0 and trim_end is None:
        return sound
    try:
        return SoundUtil.trim(sound, trim_start, trim_end)
    except ValueError as error:
        raise WebRenderError(f"Choose a trim range within the {sound.duration:.2f}-second reference clip.") from error


def preview_reference(path: str, *, trim_start: float = 0.0, trim_end: float | None = None) -> str:
    """Write the selected part of the reference audio as a playable WAV."""
    sound = _load_reference_audio(path, trim_start, trim_end)
    if not sound.data.size:
        raise WebRenderError("The reference audio is empty.")
    try:
        return _write_preview_wav(sound, "tts-reference-")
    except (OSError, RuntimeError, ValueError) as error:
        raise WebRenderError("The reference audio preview could not be prepared.") from error


def transcribe_reference(
    path: str, *, language_code: str = "en", trim_start: float = 0.0, trim_end: float | None = None,
) -> str:
    """Transcribe a reference clip using the local CPU Whisper cache."""
    from faster_whisper.utils import download_model
    from huggingface_hub.errors import LocalEntryNotFoundError

    from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
    from tts_audiobook_tool.stt import FasterWhisperAdapter
    from tts_audiobook_tool.text_ops.language_util import normalize_language_code
    from tts_audiobook_tool.transcriber import Transcriber

    with _render_job(None):
        sound = _load_reference_audio(path, trim_start, trim_end)
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
    default_voice = RenderVoice()
    narrator_voice = voices.get(narrator_id, voices.get(_NARRATOR, default_voice))
    speakers = session.attributions.speaker_of_each_line(session.book)
    book = deepcopy(session.book)
    references = [narrator_voice]
    voice_indices = {narrator_voice: 0}
    for index, group in enumerate(book.phrase_groups):
        settings = session.line_voices.get(group.voice_index) if group.voice_index > 0 else None
        if settings is None:
            speaker = canonical_cast_role(speakers.get(index, narrator_id))
            settings = voices.get(speaker, default_voice)
            if settings == default_voice:
                settings = narrator_voice
        if settings not in voice_indices:
            voice_indices[settings] = len(references)
            references.append(settings)
        group.voice_index = voice_indices[settings]
    return book, references


def _cast_voice_labels(session: BookSession, cast: dict[str, RenderVoice], references: list[RenderVoice]) -> list[str]:
    characters = session.roster.characters if session.roster else ()
    character_names = {character.id: character.name for character in characters}
    character_names.update(UNIDENTIFIED_CAST_ROLES)
    roles_by_voice: dict[RenderVoice, list[str]] = {}
    for role, voice in cast.items():
        roles_by_voice.setdefault(voice, []).append(character_names.get(canonical_cast_role(role), role))
    labels = []
    for index, voice in enumerate(references):
        fallback = "Narrator" if index == 0 else " / ".join(roles_by_voice.get(voice, ()))
        labels.append(_voice_label(index, voice.name.strip() or fallback))
    return labels


def _checkpoint_id(
    book: Book,
    references: list[RenderVoice],
    model_id: str,
    language_code: str,
    validate_audio: bool,
    batch_size: int,
    *,
    model_options: dict[str, object] | None = None,
    word_substitutions: dict[str, str] | None = None,
    strictness: Strictness | None = None,
    max_retries: int = PROJECT_MAX_RETRIES_DEFAULT,
    runtime_settings: WebRuntimeSettings | None = None,
    remote_selection: RemoteTtsSelection | None = None,
    voice_labels: list[str] | None = None,
) -> str:
    voice_settings = []
    defaults = asdict(RenderVoice())
    uses_builtin_presets = model_id == "pocket_local" or (
        model_id == "qwen3tts_local" and (model_options or {}).get("model_type") == "custom_voice"
    )
    for index, voice in enumerate(references):
        clip_digest = ""
        has_active_preset = uses_builtin_presets and model_id in dict(voice.presets)
        if voice.reference_path and not has_active_preset:
            try:
                with Path(voice.reference_path).open("rb") as clip:
                    clip_digest = hashlib.file_digest(clip, "sha256").hexdigest()
            except OSError as error:
                label = voice_labels[index] if voice_labels is not None else _voice_label(index, voice.name)
                raise WebRenderError(f"{label} reference audio is unavailable.") from error
        settings = asdict(voice)
        del settings["reference_path"]
        voice_key = [clip_digest, settings.pop("transcript").strip(), settings.pop("instructions").strip()]
        # Keep existing checkpoints addressable for voices from older web projects.
        extra = {field: value for field, value in settings.items() if value != defaults[field]}
        if extra:
            voice_key.append(extra)
        voice_settings.append(voice_key)

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
    if model_options:
        payload["model_options"] = model_options
    if word_substitutions:
        payload["word_substitutions"] = list(word_substitutions.items())
    if strictness is not None and strictness is not Strictness.get_recommended_default(language_code):
        payload["strictness"] = strictness.id
    if max_retries != PROJECT_MAX_RETRIES_DEFAULT:
        payload["max_retries"] = max_retries
    if runtime_settings is not None and (runtime := runtime_settings.checkpoint_options(validate_audio=validate_audio)):
        payload["runtime_settings"] = runtime
    if remote_selection is not None:
        payload["remote_selection"] = remote_selection.key
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_complete_audio(path: Path) -> bool:
    import soundfile

    try:
        with soundfile.SoundFile(path) as audio:
            audio.seek(-1, SEEK_END)
            return bool(audio.read(1).size)
    except (OSError, ValueError, soundfile.LibsndfileError):
        return False


def _missing_segments(project: Project, indices: Collection[int]) -> set[int]:
    """Scan requested takes, discarding files left incomplete by an interrupted write."""
    from tts_audiobook_tool.project_support.sound_segment_util import get_segment_stt_info_path

    project.sound_segments.force_invalidate()
    missing = set()
    for index in indices:
        while file_name := project.sound_segments.get_best_file_for(index):
            path = Path(project.sound_segments_path) / file_name
            if _has_complete_audio(path):
                break
            L.w(f"Discarding unreadable segment {file_name} for resumed generation")
            path.unlink(missing_ok=True)
            get_segment_stt_info_path(path).unlink(missing_ok=True)
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
    on_telemetry: Callable[[BookRenderTelemetry], None] | None = None,
    cancel_event: threading.Event | None = None,
    model_options: dict[str, object] | None = None,
    first_line: int = 1,
    last_line: int | None = None,
    regenerate: bool = False,
    auto_concat: bool = True,
    strictness: str | None = None,
    max_retries: int = PROJECT_MAX_RETRIES_DEFAULT,
    runtime_settings: WebRuntimeSettings | None = None,
    remote_selection: RemoteTtsSelection | None = None,
    on_console: Callable[[str], None] | None = None,
) -> str | None:
    """Generate a requested range and export when the whole book is ready."""
    from filelock import FileLock, Timeout

    line_count = len(session.book.phrase_groups)
    if not line_count:
        raise WebRenderError("Import book text first.")
    last_line = line_count if last_line is None else last_line
    if type(first_line) is not int or type(last_line) is not int or not 1 <= first_line <= last_line <= line_count:
        raise WebRenderError(f"Choose an inclusive line range from 1 to {line_count}.")
    if type(batch_size) is not int or not 1 <= batch_size <= PROJECT_BATCH_SIZE_MAX:
        raise WebRenderError(f"Choose a batch size from 1 to {PROJECT_BATCH_SIZE_MAX}.")
    tolerance = Strictness.get_by_id(strictness)
    if strictness is not None and tolerance is None:
        raise WebRenderError("Choose a word error tolerance.")
    if type(max_retries) is not int or not PROJECT_MAX_RETRIES_MIN <= max_retries <= PROJECT_MAX_RETRIES_MAX:
        raise WebRenderError(f"Choose a retry count from {PROJECT_MAX_RETRIES_MIN} to {PROJECT_MAX_RETRIES_MAX}.")
    if type(regenerate) is not bool or type(auto_concat) is not bool:
        raise WebRenderError("Choose valid generation and export settings.")
    runtime_settings = runtime_settings or WebRuntimeSettings()
    selected = set(range(first_line - 1, last_line))
    with _render_job(cancel_event):
        model = TtsModelType.require_by_id(remote_selection.catalog_id) if remote_selection else _detect_model()
        options = _validated_model_options(model.id, model_options)
        if not model.can_batch():
            batch_size = 1
        book, references = _cast_book(session, voices)
        language_code = session.language_code if language_code is None else language_code
        voice_labels = _cast_voice_labels(session, voices, references)
        if on_console is not None:
            on_console(f"Audiobook lines {first_line}–{last_line} of {line_count} (inclusive). Batch size {batch_size}.\n")
            validation = "off"
            if validate_audio:
                variant = runtime_settings.whisper_variant or SttVariant.get_default().id
                device = runtime_settings.whisper_device or "automatic device"
                validation = f"on ({variant}, {device})"
            effective_tolerance = tolerance or Strictness.get_recommended_default(language_code)
            on_console(
                f"Whisper validation {validation}. "
                f"Word error tolerance {effective_tolerance.label.lower()}. Max retries {max_retries}.\n"
            )
        checkpoint = _checkpoint_id(
            book, references, model.id, language_code, validate_audio, batch_size, model_options=options,
            word_substitutions=session.word_substitutions, strictness=tolerance, max_retries=max_retries,
            runtime_settings=runtime_settings,
            remote_selection=remote_selection,
            voice_labels=voice_labels,
        )
        work = Path(session.project_dir) / "renders" / checkpoint
        work.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(work / ".render.lock"), timeout=0)
        try:
            with lock:
                return _render_book_checkpoint(
                    work, book, references, language_code, validate_audio,
                    batch_size, on_progress, cancel_event, model=model, model_options=options,
                    on_telemetry=on_telemetry,
                    word_substitutions=session.word_substitutions, selected=selected, regenerate=regenerate,
                    auto_concat=auto_concat, strictness=tolerance, max_retries=max_retries,
                    runtime_settings=runtime_settings,
                    remote_selection=remote_selection,
                    voice_labels=voice_labels,
                    on_console=on_console,
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
    *,
    on_telemetry: Callable[[BookRenderTelemetry], None] | None,
    model: TtsModelType,
    model_options: dict[str, object],
    word_substitutions: dict[str, str],
    selected: set[int],
    regenerate: bool,
    auto_concat: bool,
    strictness: Strictness | None,
    max_retries: int,
    runtime_settings: WebRuntimeSettings,
    voice_labels: list[str],
    remote_selection: RemoteTtsSelection | None = None,
    on_console: Callable[[str], None] | None = None,
) -> str | None:
    _check_cancelled(cancel_event)
    total = len(selected)
    processed = 0
    synthesis_seconds = 0.0
    audio_seconds = 0.0
    eta_seconds = None
    previous_telemetry = None

    def report_telemetry() -> None:
        nonlocal previous_telemetry
        if on_telemetry is not None:
            rate = audio_seconds / synthesis_seconds if synthesis_seconds else None
            telemetry = BookRenderTelemetry(processed, total, rate, eta_seconds if processed < total else None)
            if telemetry != previous_telemetry:
                on_telemetry(telemetry)
                previous_telemetry = telemetry

    def report_progress(completed: int) -> None:
        nonlocal processed
        processed = completed
        if on_progress is not None:
            on_progress(processed, total)
        report_telemetry()

    output = work / "audiobook.abr.m4b"
    if auto_concat and not regenerate and _automatic_export_matches(output, runtime_settings.aac_bitrate):
        if on_console is not None:
            on_console("Reusing the completed M4B export. No segments need rendering.\n")
        report_progress(total)
        return str(output)

    if on_console is not None:
        voice_indices = {book.phrase_groups[index].voice_index or 0 for index in selected}
        checked_voices = ", ".join(voice_labels[index] for index in sorted(voice_indices))
        on_console(f"Checking voice references for the selected lines ({checked_voices}).\n")
    with _project_state(
        work, language_code, references, book, model=model, model_options=model_options, validate_audio=validate_audio,
        word_substitutions=word_substitutions, generation_indices=selected, strictness=strictness,
        max_retries=max_retries, runtime_settings=runtime_settings,
        remote_selection=remote_selection,
        voice_labels=voice_labels, on_console=on_console,
    ) as state:
        remaining = selected if regenerate else _missing_segments(state.project, selected)
        if on_console is not None:
            on_console(f"Selected segments include {total - len(remaining)} reused and {len(remaining)} to render.\n")
        report_progress(total - len(remaining))
        _check_cancelled(cancel_event)
        if remaining:
            try:
                runtime_settings.check_validation_availability(state.prefs)
            except WebRuntimeSettingsError as error:
                raise WebRenderError(str(error)) from error

            def report_generation_progress(update: GenerationProgress) -> None:
                nonlocal eta_seconds
                eta_seconds = update.eta_seconds
                report_progress(total - update.total + update.processed)

            def report_generation_stats(update: GenerationStats) -> None:
                nonlocal synthesis_seconds, audio_seconds
                if update.generation_seconds > 0 and update.audio_seconds > 0:
                    synthesis_seconds += update.generation_seconds
                    audio_seconds += update.audio_seconds
                    report_telemetry()

            staging = tempfile.TemporaryDirectory(dir=work, prefix=".regeneration-") if regenerate else nullcontext("")
            with staging as staging_dir, _generation_log(work, state.prefs.save_gen_log) as transcript:
                try:
                    operation_id = ModelWorker.submit_generation(
                        state=state, indices=remaining, batch_size=batch_size, is_regen=regenerate, staging_dir=staging_dir,
                        suppress_validation_passes=True,
                    )
                except ModelWorkerUnavailable as error:
                    L.w(str(error))
                    raise WebRenderError("The model worker could not start.") from error
                _wait_for_job(
                    operation_id, GenerationFinished, cancel_event, report_generation_progress,
                    on_stats=report_generation_stats if on_telemetry is not None else None,
                    transcript=transcript, on_console=on_console,
                )
                _check_cancelled(cancel_event)
                if regenerate:
                    _publish_regenerated_takes(state.project, staging_dir, selected, output)
                if missing := _missing_segments(state.project, selected):
                    raise WebRenderError(
                        f"The model did not produce audio for {len(missing)} of {total} text segments."
                    )
            if on_console is not None:
                on_console("Generation completed.\n")
            report_progress(total)

        _check_cancelled(cancel_event)
        if not auto_concat:
            if on_console is not None:
                on_console("Automatic export is off.\n")
            return None
        if unrendered := _missing_segments(state.project, set(range(len(book.phrase_groups))) - selected):
            if on_console is not None:
                on_console(f"Automatic export pending. {len(unrendered)} segments elsewhere in the book need audio.\n")
            return None
        if on_console is not None:
            on_console("Proceeding to concatenation...\n")
        return _export_checkpoint(state, output, len(book.phrase_groups), cancel_event)


@contextmanager
def _generation_log(project_dir: Path, enabled: bool) -> Iterator[GenerationTranscript | None]:
    if not enabled:
        yield None
        return

    from tts_audiobook_tool.textual.generation_app import GenerationTranscript, make_generation_transcript_path

    try:
        transcript = GenerationTranscript(make_generation_transcript_path(str(project_dir)))
    except OSError as error:
        raise WebRenderError("The generation log could not be opened.") from error
    try:
        yield transcript
    except WebRenderError as error:
        transcript.write_lines([str(error)])
        raise
    else:
        transcript.write_lines(["Generation completed."])
    finally:
        transcript.close()


def _publish_regenerated_takes(project: Project, staging_dir: str, selected: set[int], output: Path) -> None:
    from tts_audiobook_tool.project_support.segment_staging_util import list_staged_segments
    from tts_audiobook_tool.project_support.sound_segment_util import (
        SoundSegmentUtil, get_segment_error_rank, get_segment_stt_info_path,
    )

    best = {}
    for path, segment in list_staged_segments(staging_dir):
        if segment.idx not in selected or path.is_symlink() or not _has_complete_audio(path):
            continue
        if segment.hash != SoundSegmentUtil.calc_segment_hash(segment.idx, project.phrase_groups[segment.idx].text):
            continue
        rank = get_segment_error_rank(segment, staging_dir)
        if segment.idx not in best or rank < best[segment.idx][0]:
            best[segment.idx] = (rank, path)
    if len(best) != len(selected):
        missing = len(selected) - len(best)
        raise WebRenderError(f"The model did not produce new audio for {missing} of {len(selected)} text segments.")

    segments_dir = Path(project.sound_segments_path)
    previous_output = Path(staging_dir) / "previous.abr.m4b"
    published = []
    try:
        # Remove the stale export before a new take can become visible. Keep it
        # in the staging directory until publication succeeds.
        if output.is_file():
            output.replace(previous_output)
        for _, path in best.values():
            while True:
                destination = segments_dir / f"{path.stem} [take {uuid4().hex[:12]}]{path.suffix}"
                destination_sidecar = get_segment_stt_info_path(destination)
                if not destination.exists() and not destination_sidecar.exists():
                    break
            sidecar = get_segment_stt_info_path(path)
            if sidecar.is_file() and not sidecar.is_symlink():
                sidecar.replace(destination_sidecar)
                published.append(destination_sidecar)
            path.replace(destination)
            published.append(destination)
    except OSError as error:
        L.w(str(error))
        try:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            if previous_output.is_file():
                previous_output.replace(output)
        except OSError as rollback_error:
            L.w(str(rollback_error))
        raise WebRenderError("The new audio takes could not be saved.") from error
    finally:
        project.sound_segments.force_invalidate()


def _export_stamp(audio: Path, bitrate: str) -> dict[str, str | int]:
    stat = audio.stat()
    return {"bitrate": bitrate, "size": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _automatic_export_matches(output: Path, bitrate: str) -> bool:
    if output.is_symlink() or not output.is_file():
        return False
    marker = output.with_suffix(".bitrate.json")
    if marker.is_symlink():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Exports produced before the bitrate preference always used the app default.
        return bitrate == AAC_BITRATE_DEFAULT
    except (OSError, ValueError):
        return False
    try:
        return saved == _export_stamp(output, bitrate)
    except OSError:
        return False


def _export_checkpoint(state: State, output: Path, total: int, cancel_event: threading.Event | None) -> str:
    from tts_audiobook_tool.concat_util import ConcatUtil

    staging_dir = output.parent / "export"
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
    replacement = Path(path)
    prepared_marker = staging_dir / "aac-bitrate.json"
    try:
        prepared_marker.write_text(json.dumps(_export_stamp(replacement, state.prefs.aac_bitrate)), encoding="utf-8")
        # Publish the new file's stamp first. If interrupted before the audio
        # rename, the stamp cannot be mistaken for a matching old export.
        prepared_marker.replace(output.with_suffix(".bitrate.json"))
        replacement.replace(output)
    except OSError as error:
        L.w(str(error))
        raise WebRenderError("The audiobook audio could not be exported.") from error
    return str(output)
