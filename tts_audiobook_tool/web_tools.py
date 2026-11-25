"""File and audio tools for the local browser interface."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace

from natsort import natsorted, ns

from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, ABR_VERSION, APP_META_FLAC_FIELD, APP_META_MP4_MEAN, APP_META_MP4_TAG
from tts_audiobook_tool.util import progress_eta_description


AUDIO_SUFFIXES = (".mp3", ".m4a", ".m4b", ".flac", ".wav", ".ogg", ".opus", ".aac", ".aiff", ".aif", ".webm", ".mp4")
_COPY_SUFFIXES = (".m4a", ".m4b")
_MEDIA_SIGNATURES = {
    ".flac": (b"fLaC",),
    ".ogg": (b"OggS",),
    ".opus": (b"OggS",),
    ".aif": (b"FORM",),
    ".aiff": (b"FORM",),
    ".webm": (b"\x1a\x45\xdf\xa3",),
}


class WebToolsError(ValueError):
    """A browser-safe message for a failed tool operation."""


@dataclass(frozen=True)
class ToolResult:
    path: Path
    message: str


def _uploaded_file(path: str | None, upload_root: str | Path, suffixes: Sequence[str]) -> Path:
    source = Path(path or "")
    try:
        if source.suffix.lower() not in suffixes or source.is_symlink() or not source.is_file():
            raise OSError
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(Path(upload_root).resolve(strict=True)):
            raise OSError
    except (OSError, RuntimeError, ValueError):
        raise WebToolsError(f"Upload a file in a supported format ({', '.join(suffixes)}).") from None
    return resolved


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise WebToolsError("Stopped.")


def _uploaded_audio(path: str | None, upload_root: str | Path, suffixes: Sequence[str] = AUDIO_SUFFIXES) -> Path:
    source = _uploaded_file(path, upload_root, suffixes)
    try:
        with source.open("rb") as opened:
            header = opened.read(12)
    except OSError:
        raise WebToolsError("Could not read the uploaded audio.") from None
    # The shared decoder auto-detects formats. Restrict browser uploads to
    # media containers before it opens them, so a renamed playlist cannot load.
    suffix = source.suffix.lower()
    if suffix in _MEDIA_SIGNATURES:
        valid = header.startswith(_MEDIA_SIGNATURES[suffix])
    elif suffix == ".wav":
        valid = header[:4] in (b"RIFF", b"RIFX", b"RF64", b"BW64") and header[8:12] == b"WAVE"
    elif suffix in (".mp3", ".aac"):
        valid = header.startswith((b"ID3", b"ADIF")) or (len(header) >= 2 and header[0] == 255 and header[1] >= 224)
    else:
        valid = header[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide")
    if not valid:
        raise WebToolsError("The uploaded audio does not match its file format.")
    return source


def epub_to_text(path: str, output_dir: Path, *, upload_root: str | Path) -> ToolResult:
    from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor

    source = _uploaded_file(path, upload_root, (".epub",))
    try:
        extracted = EpubExtractor.extract_import_chapters(str(source))
        raw_text = "\n\n".join(chapter.text for chapter in extracted.chapters)
    except Exception:
        raise WebToolsError("Could not read the EPUB.") from None
    if not raw_text.strip():
        raise WebToolsError("The EPUB contains no readable text.")
    result = output_dir / source.with_suffix(".txt").name
    result.write_text(raw_text, encoding="utf-8")
    return ToolResult(result, "\n".join(["Text file ready.", *extracted.significant_warnings]))


def _report_work(
    description: str, completed: float, total: float, started: float,
    on_progress: Callable[[str], None] | None, on_counted_progress: Callable[[str, float], None] | None,
) -> None:
    description = progress_eta_description(description, completed, total, monotonic() - started)
    if on_counted_progress is not None:
        on_counted_progress(description, min(1.0, completed / total))
    elif on_progress is not None:
        on_progress(description)


def concatenate_audio(
        paths: Sequence[str] | None, output_format: str, output_dir: Path, *, upload_root: str | Path,
        cancel_event: Event | None = None, on_progress: Callable[[str], None] | None = None,
        on_counted_progress: Callable[[str, float], None] | None = None,
) -> ToolResult:
    from tts_audiobook_tool.sound.audio_convert import AudioConvertCancelled, convert_audio_files

    if output_format not in ("flac", "m4a"):
        raise WebToolsError("Choose FLAC or M4A output.")
    if not paths:
        raise WebToolsError("Upload at least one audio file.")
    sources = [_uploaded_audio(path, upload_root) for path in paths]
    sources = natsorted(sources, key=lambda path: path.name, alg=ns.IGNORECASE)
    result = output_dir / f"concatenated.{output_format}"

    started = monotonic()

    def report(index: int, path: str) -> None:
        _report_work(
            f"Combining {index + 1} of {len(sources)}: {Path(path).name}", index, len(sources), started,
            on_progress, on_counted_progress,
        )

    _check_cancelled(cancel_event)
    try:
        convert_audio_files(
            [str(path) for path in sources], str(result), on_source=report,
            cancel_check=cancel_event.is_set if cancel_event is not None else None,
        )
    except AudioConvertCancelled:
        raise WebToolsError("Stopped.") from None
    except Exception:
        raise WebToolsError("Could not combine the audio. Check that every file is playable.") from None
    return ToolResult(result, f"Audio ready. Processed {len(sources)} {'file' if len(sources) == 1 else 'files'}.")


def transcode_audiobook(
        path: str, output_dir: Path, *, upload_root: str | Path, cancel_event: Event | None = None,
) -> ToolResult:
    from tts_audiobook_tool.sound.audio_convert import AudioConvertCancelled, transcode_to_aac
    from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil

    source = _uploaded_audio(path, upload_root, (".flac",))
    _check_cancelled(cancel_event)
    metadata = AudioMetaUtil.get_flac_metadata_field(str(source), APP_META_FLAC_FIELD)
    if not metadata:
        raise WebToolsError("This FLAC has no synchronized text metadata from the app.")
    result = output_dir / source.with_suffix(".m4b").name
    try:
        transcode_to_aac(
            str(source), str(result), bitrate=AAC_BITRATE_DEFAULT,
            cancel_check=cancel_event.is_set if cancel_event is not None else None,
        )
        _check_cancelled(cancel_event)
        if AudioMetaUtil.set_mp4_metadata_tag(str(result), APP_META_MP4_MEAN, APP_META_MP4_TAG, metadata):
            raise WebToolsError("Could not save the synchronized text metadata in the M4B.")
    except AudioConvertCancelled:
        raise WebToolsError("Stopped.") from None
    except WebToolsError:
        result.unlink(missing_ok=True)
        raise
    except Exception:
        result.unlink(missing_ok=True)
        raise WebToolsError("Could not transcode the audiobook.") from None
    return ToolResult(result, "M4B ready with the original synchronized text metadata.")


def change_reference_speed(path: str, percent: int, output_dir: Path, *, upload_root: str | Path) -> ToolResult:
    from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
    from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil

    if type(percent) not in (int, float) or not 50 <= percent <= 200 or percent != int(percent):
        raise WebToolsError("Choose a whole-number speed from 50% to 200%.")
    source = _uploaded_audio(path, upload_root)
    sound = SoundFileUtil.load(str(source))
    if isinstance(sound, str) or not sound.data.size:
        raise WebToolsError("Could not read any audio in the reference clip.")
    adjusted = SoundExtraUtil.speed_up_audio(sound, percent / 100) if percent != 100 else sound
    if isinstance(adjusted, str) or not adjusted.data.size:
        raise WebToolsError("Could not change the speed. Try a longer reference clip.")
    result = output_dir / f"{source.stem}_speed_{int(percent)}_percent.flac"
    if SoundFileUtil.save_flac(adjusted, str(result)):
        raise WebToolsError("Could not save the adjusted clip.")
    return ToolResult(result, f"Reference clip ready at {int(percent)}% speed.")


def enhance_audiobook(
        path: str, source_text: str, language: str, output_dir: Path, *, upload_root: str | Path,
        cancel_event: Event | None = None, on_progress: Callable[[str], None] | None = None,
        on_counted_progress: Callable[[str, float], None] | None = None,
) -> ToolResult:
    from tts_audiobook_tool.app_types.app_metadata import AppMetadata
    from tts_audiobook_tool.enhance import enhance_text
    from tts_audiobook_tool.enhance.enhance_artifacts import count_misalignments
    from tts_audiobook_tool.sound.audio_convert import AudioConvertCancelled, transcode_to_aac
    from tts_audiobook_tool.text_ops.language_util import normalize_language_code

    audio = _uploaded_audio(path, upload_root)
    text = _uploaded_file(source_text, upload_root, (".txt", ".epub"))
    code = normalize_language_code(language or "") or "en"
    _check_cancelled(cancel_event)
    if on_progress is not None:
        on_progress("Reading the source text")
    book = _read_enhance_book(text, code)
    words = _transcribe_cached(audio, code, cancel_event, on_progress, on_counted_progress)
    if not words:
        raise WebToolsError("Whisper found no speech in the audiobook.")
    _check_cancelled(cancel_event)
    if on_progress is not None:
        on_progress("Aligning text with the audiobook")
    timed, interrupted = enhance_text.align_book(book, words)
    if interrupted:
        raise WebToolsError("Stopped.")
    _check_cancelled(cancel_event)
    unmatched = count_misalignments(timed)
    if not timed or unmatched == len(timed):
        raise WebToolsError("The source text could not be aligned with the audiobook.")

    result = output_dir / (audio.stem + (".abr.m4b" if text.suffix.lower() == ".epub" else ".abr.m4a"))
    metadata = AppMetadata(
        timed_phrases=list(timed), title=book.title or audio.stem, version=ABR_VERSION,
        bookmark_indices=[], raw_text="", has_break_audio=False, project_snapshot={},
        sections=enhance_text.make_app_metadata_sections(book, len(timed)), type="conversion",
    )
    if on_progress is not None:
        on_progress("Writing the synchronized audiobook")
    try:
        if audio.suffix.lower() in _COPY_SUFFIXES:
            error = AppMetadata.save_to_mp4(metadata, str(audio), str(result))
        else:
            transcode_to_aac(
                str(audio), str(result), bitrate=AAC_BITRATE_DEFAULT,
                cancel_check=cancel_event.is_set if cancel_event is not None else None,
            )
            error = AppMetadata.save_to_mp4(metadata, str(result))
        _check_cancelled(cancel_event)
        if error or AppMetadata.load_from_file(str(result)) is None:
            raise WebToolsError("Could not verify the synchronized text in the output audiobook.")
    except AudioConvertCancelled:
        raise WebToolsError("Stopped.") from None
    except WebToolsError:
        result.unlink(missing_ok=True)
        raise
    except Exception:
        result.unlink(missing_ok=True)
        raise WebToolsError("Could not create the enhanced audiobook.") from None
    matched = len(timed) - unmatched
    return ToolResult(result, f"Audiobook ready. Aligned {matched:,} of {len(timed):,} text segments.")


def _read_enhance_book(path: Path, language: str):
    from tts_audiobook_tool.constants_config import (
        ENHANCE_DIALOG_SEGMENTATION, ENHANCE_MAX_WORDS_PER_SEGMENT, ENHANCE_SEGMENTATION_STRATEGY,
    )
    from tts_audiobook_tool.enhance import enhance_text
    from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor

    settings = SimpleNamespace(project=SimpleNamespace(language_code=language))
    try:
        if path.suffix.lower() == ".txt":
            book = enhance_text.plain_text_to_book(path.read_text(encoding="utf-8-sig"), settings)
        else:
            extracted = EpubExtractor.import_epub(
                str(path), ENHANCE_MAX_WORDS_PER_SEGMENT, ENHANCE_SEGMENTATION_STRATEGY, language,
                dialog_segmentation=ENHANCE_DIALOG_SEGMENTATION,
            )
            book = enhance_text.epub_result_to_book(extracted, settings)
    except Exception:
        raise WebToolsError("Could not read the source text. Use an EPUB or UTF-8 text file.") from None
    if isinstance(book, str):
        raise WebToolsError("Could not find readable text segments in the source file.")
    return book


def _transcribe_cached(audio: Path, language: str, cancel_event: Event | None, on_progress, on_counted_progress):
    from faster_whisper.utils import download_model

    from tts_audiobook_tool.app_types import ConcreteWord, SttConfig, SttVariant
    from tts_audiobook_tool.constants_config import ENHANCE_OVERLAP_STITCHING
    from tts_audiobook_tool.enhance import enhance_alignment as alignment
    from tts_audiobook_tool.stt import FasterWhisperAdapter
    from tts_audiobook_tool.transcriber import Transcriber

    _check_cancelled(cancel_event)
    try:
        snapshot = Path(download_model(SttVariant.get_default().id, local_files_only=True))
    except Exception:
        raise WebToolsError("Enhancement requires the default Whisper model in the local cache.") from None
    if not all((snapshot / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")):
        raise WebToolsError("The cached Whisper model is incomplete.")
    if on_progress is not None:
        on_progress("Loading cached Whisper on the CPU")
    config = SttConfig.CPU_INT8FLOAT32
    chunk = alignment.TRANSCRIBE_CHUNK_SECONDS
    overlap = alignment.TRANSCRIBE_OVERLAP_SECONDS
    stride = chunk - overlap
    transcript = []
    try:
        whisper = FasterWhisperAdapter(
            str(snapshot), device=config.device, compute_type=config.compute_type, cpu_threads=4,
        )
        audio_duration: float | None = None

        def set_duration(duration: float | None) -> None:
            nonlocal audio_duration
            audio_duration = duration

        windows = alignment._stream_audio_with_overlap(
            str(audio), chunk, overlap, cancel_check=cancel_event.is_set if cancel_event is not None else None,
            on_duration=set_duration if on_progress or on_counted_progress else None,
        )
        started = monotonic()
        with closing(windows):
            for index, window in enumerate(windows):
                _check_cancelled(cancel_event)
                offset = index * stride
                description = f"Transcribing audio at {offset:,} seconds"
                if audio_duration and audio_duration > offset:
                    _report_work(description, offset, audio_duration, started, on_progress, on_counted_progress)
                elif on_progress is not None:
                    on_progress(description)
                segments, _ = whisper.transcribe(
                    window, language=language if language in whisper.supported_languages else None,
                    word_timestamps=True,
                )
                transcript.append([
                    ConcreteWord(word.start + offset, word.end + offset, word.word, word.probability)
                    for word in Transcriber.get_words_from_segments(segments)
                ])
        _check_cancelled(cancel_event)
    except WebToolsError:
        raise
    except Exception:
        raise WebToolsError("Could not transcribe the audiobook with cached Whisper.") from None
    if ENHANCE_OVERLAP_STITCHING:
        return alignment._stitch_transcripts_by_overlap(transcript, stride=stride, overlap=overlap)
    return alignment._stitch_transcripts(transcript)
