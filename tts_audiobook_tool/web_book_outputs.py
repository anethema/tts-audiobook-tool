"""Review takes and export audio from an existing browser rendering project."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from filelock import FileLock, Timeout

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types import (
    ExportType, HighShelfEq, MinSegmentSentenceSilence, NormalizationType, RoomReverbStereo, RoomTone, SectionMarkerMode,
)
from tts_audiobook_tool.constants import (
    AAC_BITRATE_DEFAULT, AAC_BITRATES, PROJECT_JSON_FILE_NAME, PROJECT_SOUND_SEGMENTS_SUBDIR, PROJECT_TEXT_FILE_NAME,
    PROJECT_TEXT_RAW_FILE_NAME,
)
from tts_audiobook_tool.constants_config import SILENCE_GAP_DURATION_MAX, SILENCE_GAP_DURATION_MIN
from tts_audiobook_tool.l import L
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project_support.project_load_util import ProjectLoadUtil
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.project_support.sound_segment_util import UNKNOWN_ERROR_RANK, get_segment_error_rank
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.checkpoint_audio import has_complete_audio
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import make_file_line_ranges

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.state import State
    from tts_audiobook_tool.web_backend import BookSession


_RUN_ID = re.compile(r"[0-9a-f]{64}")


class WebBookOutputError(ValueError):
    """An audio review or export error suitable for the browser."""


@dataclass(frozen=True)
class BookRenderRun:
    run_id: str
    title: str
    model_id: str
    model_label: str
    generated_lines: int
    total_lines: int
    updated_at: float
    performance_cues: bool = False


@dataclass(frozen=True)
class BookTake:
    filename: str
    word_errors: int | None
    failed: bool
    preferred: bool


@dataclass(frozen=True)
class BookLineReview:
    line: int
    text: str
    takes: tuple[BookTake, ...]


@dataclass(frozen=True)
class BookExportOptions:
    export_type: ExportType = ExportType.AAC
    aac_bitrate: str = AAC_BITRATE_DEFAULT
    reason_pauses: ReasonPauseTypes | None = None
    min_segment_sentence_silence: MinSegmentSentenceSilence | None = None
    limit_silence_gaps: bool | None = None
    limit_silence_gaps_duration: float | None = None
    use_break_sound_effect: bool | None = None
    subdivide_phrases: bool | None = None
    normalization_type: NormalizationType | None = None
    high_shelf: HighShelfEq | None = None
    room_tone: RoomTone | None = None
    room_reverb_stereo: RoomReverbStereo | None = None
    use_upsampler: bool = False


@dataclass(frozen=True)
class BookExport:
    path: str
    line_start: int
    line_end: int
    missing_lines: int
    export_id: str = ""


def _runs_root(session: BookSession) -> Path:
    return Path(session.project_dir).resolve() / "renders"


def _run_directory(session: BookSession, run_id: str) -> Path:
    root = _runs_root(session)
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None or root.is_symlink():
        raise WebBookOutputError("Choose a rendering from this project.")
    directory = root / run_id
    if directory.is_symlink() or not directory.is_dir():
        raise WebBookOutputError("That rendering is unavailable.")
    return directory


@contextmanager
def _open_render_project(directory: Path) -> Iterator[Project]:
    if not hasattr(L, "logger"):
        L.init("tts-audiobook-web")
    for name in (PROJECT_JSON_FILE_NAME, PROJECT_TEXT_FILE_NAME):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise WebBookOutputError("This rendering’s project files are unavailable.")
    if (directory / PROJECT_TEXT_RAW_FILE_NAME).is_symlink():
        raise WebBookOutputError("This rendering’s project files are unavailable.")
    segments = directory / PROJECT_SOUND_SEGMENTS_SUBDIR
    if segments.is_symlink() or not segments.is_dir() or any(path.is_symlink() for path in segments.iterdir()):
        raise WebBookOutputError("This rendering’s generated audio folder is unavailable.")
    try:
        project = ProjectLoadUtil.load_using_dir_path(str(directory), prompt_on_warnings=False)
    except (OSError, ValueError) as error:
        L.w(str(error))
        raise WebBookOutputError("This rendering’s project could not be opened.") from error
    if isinstance(project, str):
        L.w(project)
        raise WebBookOutputError("This rendering’s project could not be opened.")
    try:
        project.sound_segments.dont_show_scan_message = True
        yield project
    finally:
        project.kill()
        project.sound_segments.observer.join()


def list_book_render_runs(session: BookSession) -> tuple[BookRenderRun, ...]:
    """Find saved rendering projects, including stopped or incomplete runs."""
    root = _runs_root(session)
    if root.is_symlink() or not root.is_dir():
        return ()
    runs = []
    for directory in root.iterdir():
        if not _RUN_ID.fullmatch(directory.name) or not directory.is_dir() or directory.is_symlink():
            continue
        try:
            with _open_render_project(directory) as project:
                model = TtsModelType.get_by_id(project.tts_model_type)
                runs.append(BookRenderRun(
                    directory.name, project.book.title, project.tts_model_type,
                    (model.value.ui.get("proper_name") if model.id != "none" else None) or project.tts_model_type,
                    project.sound_segments.num_generated(), len(project.phrase_groups),
                    (directory / PROJECT_SOUND_SEGMENTS_SUBDIR).stat().st_mtime,
                    performance_cues=any(group.delivery for group in project.phrase_groups),
                ))
        except (OSError, WebBookOutputError):
            continue
    return tuple(sorted(runs, key=lambda run: (run.updated_at, run.run_id), reverse=True))


def _line_index(line: int, count: int) -> int:
    if type(line) is not int or not 1 <= line <= count:
        raise WebBookOutputError(f"Choose a line from 1 to {count}.")
    return line - 1


def review_book_line(session: BookSession, run_id: str, line: int) -> BookLineReview:
    """Read the stored text and rank takes using the console's validation score."""
    with _open_render_project(_run_directory(session, run_id)) as project:
        index = _line_index(line, len(project.phrase_groups))
        best = project.sound_segments.get_best_item_for(index)
        takes = []
        for item in project.sound_segments.sound_segments_map.get(index, []):
            rank = get_segment_error_rank(item, project.sound_segments_path)
            takes.append(BookTake(
                item.file_name, None if rank == UNKNOWN_ERROR_RANK and item.num_errors < 0 else rank,
                project.sound_segments.is_segment_failed(index, item), item == best,
            ))
        takes.sort(key=lambda take: (not take.preferred, take.word_errors is None, take.word_errors or 0))
        return BookLineReview(line, project.phrase_groups[index].presentable_text, tuple(takes))


def get_book_take_path(session: BookSession, run_id: str, line: int, filename: str) -> str:
    """Resolve a selected take only if it belongs to the chosen saved line."""
    directory = _run_directory(session, run_id)
    with _open_render_project(directory) as project:
        index = _line_index(line, len(project.phrase_groups))
        if filename not in project.sound_segments.get_filenames_for(index):
            raise WebBookOutputError("Choose an available take for this line.")
        path = directory / PROJECT_SOUND_SEGMENTS_SUBDIR / filename
        try:
            if not has_complete_audio(path):
                raise ValueError
        except (OSError, ValueError) as error:
            raise WebBookOutputError("The selected take could not be read.") from error
        return str(path)


def _resolved_export(directory: Path, path: Path) -> Path | None:
    try:
        if path.is_relative_to(directory) and path.resolve(strict=True) == path and path.is_file():
            return path
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def list_book_exports(session: BookSession, run_id: str) -> tuple[tuple[str, str], ...]:
    """Return saved export basenames and relative IDs without copying audio."""
    directory = _run_directory(session, run_id)
    default = _resolved_export(directory, directory / "audiobook.abr.m4b")
    files = [] if default is None else [(default.name, default.name)]
    with _open_render_project(directory) as project:
        if Path(project.concat_path).is_symlink():
            return tuple(files)
        for filename, timestamp in ProjectUtil.get_latest_concat_files(project, limit=sys.maxsize):
            if path := _resolved_export(directory, Path(filename)):
                files.append((f"{path.name} · {timestamp}", path.relative_to(directory).as_posix()))
    return tuple(files)


def get_book_export_path(session: BookSession, run_id: str, export_id: str) -> str:
    directory = _run_directory(session, run_id)
    if export_id not in {value for _, value in list_book_exports(session, run_id)}:
        raise WebBookOutputError("Choose an available audiobook export.")
    path = _resolved_export(directory, directory / export_id)
    if path is None:
        raise WebBookOutputError("That audiobook export is unavailable.")
    return str(path)


def _apply_export_options(project: Project, options: BookExportOptions) -> None:
    project.export_type = options.export_type
    if options.reason_pauses is not None:
        project.reason_pauses = options.reason_pauses.value
    if options.high_shelf is not None:
        project.high_shelf = options.high_shelf.id
    for field in (
        "min_segment_sentence_silence", "limit_silence_gaps", "limit_silence_gaps_duration",
        "use_break_sound_effect", "subdivide_phrases", "normalization_type", "room_tone", "room_reverb_stereo",
    ):
        value = getattr(options, field)
        if value is not None:
            setattr(project, field, value)


def export_book_audio(
    session: BookSession, run_id: str, *, first_line: int = 1, last_line: int | None = None,
    bookmarks: tuple[int, ...] | None = None, options: BookExportOptions = BookExportOptions(),
) -> BookExport:
    """Export an inclusive range using the console's audio pipeline and ABR metadata."""
    return export_book_audio_parts(
        session, run_id, first_line=first_line, last_line=last_line, bookmarks=bookmarks, options=options,
    )[0]


def export_book_audio_parts(
    session: BookSession, run_id: str, *, first_line: int = 1, last_line: int | None = None,
    cut_lines: tuple[int, ...] = (), split_at_sections: bool = False, bookmarks: tuple[int, ...] | None = None,
    options: BookExportOptions = BookExportOptions(),
) -> tuple[BookExport, ...]:
    """Publish one file per native cut range in a single export directory."""
    from tts_audiobook_tool.concat_util import ConcatUtil
    from tts_audiobook_tool.web_text import apply_export_formatting

    if not isinstance(options.export_type, ExportType):
        raise WebBookOutputError("Choose a file type.")
    if options.aac_bitrate not in AAC_BITRATES:
        raise WebBookOutputError("Choose a supported AAC bitrate.")
    gap = options.limit_silence_gaps_duration
    if gap is not None and (type(gap) not in (int, float) or not SILENCE_GAP_DURATION_MIN <= gap <= SILENCE_GAP_DURATION_MAX):
        raise WebBookOutputError(f"Choose a maximum silence from {SILENCE_GAP_DURATION_MIN:g} to {SILENCE_GAP_DURATION_MAX:g} seconds.")
    directory = _run_directory(session, run_id)
    lock_path = directory / ".render.lock"
    if lock_path.is_symlink():
        raise WebBookOutputError("The rendering workspace is unavailable.")
    try:
        with FileLock(str(lock_path), timeout=0), _open_render_project(directory) as project:
            apply_export_formatting(project.book, session.book)
            count = len(project.phrase_groups)
            last_line = count if last_line is None else last_line
            first_idx = _line_index(first_line, count)
            last_idx = _line_index(last_line, count)
            if first_idx > last_idx:
                raise WebBookOutputError("The end line must be at or after the start line.")
            cuts = _cut_indices(project, first_idx, last_idx, cut_lines, split_at_sections)
            bookmark_indices = _bookmark_indices(project, first_idx, last_idx, bookmarks, cuts)
            ranges = [
                (first_idx + start, first_idx + end)
                for start, end in make_file_line_ranges({index - first_idx for index in cuts}, last_idx - first_idx + 1)
            ]
            missing_per_part = []
            for start, end in ranges:
                missing = ConcatUtil.num_missing_in(project, start, end)
                if missing == end - start + 1:
                    if len(ranges) == 1:
                        raise WebBookOutputError("There is no generated audio in those lines.")
                    raise WebBookOutputError(f"There is no generated audio in part lines {start + 1:,}–{end + 1:,}.")
                missing_per_part.append(missing)

            _apply_export_options(project, options)
            project.use_upsampler = options.use_upsampler
            if cuts:
                project.chapter_mode = SectionMarkerMode.FILES
                project.markers = cuts
            # Concat consumes project and preferences. Binding a State also
            # changes the active TTS model in this browser process.
            state = cast("State", SimpleNamespace(project=project, prefs=Prefs(aac_bitrate=options.aac_bitrate)))
            parent = Path(project.concat_path)
            if parent.is_symlink():
                raise WebBookOutputError("The export folder is unavailable.")
            title = app_text.sanitize_for_filename(project.book.title)[:70].strip(" ._") or "Audiobook"
            with tempfile.TemporaryDirectory(dir=directory, prefix=".export-") as scratch:
                staged = Path(scratch)
                publication = staged / "files"
                publication.mkdir()
                destination = parent / staged.name.removeprefix(".")
                exports = []
                for part, ((start, end), missing) in enumerate(zip(ranges, missing_per_part, strict=True), 1):
                    stem = title
                    if cuts:
                        stem += f" [{part} of {len(ranges)}]"
                    if start != 0 or end != count - 1:
                        stem += f" [Lines {start + 1}-{end + 1}]"
                    path, error = _make_audio_file(
                        state, start, end, bookmark_indices, str(staged / stem), options.use_upsampler,
                    )
                    if error or _resolved_export(staged, Path(path)) is None:
                        if error:
                            L.w(error)
                        raise WebBookOutputError("The audiobook audio could not be exported.")
                    output = destination / Path(path).name
                    Path(path).rename(publication / output.name)
                    exports.append(BookExport(
                        str(output), start + 1, end + 1, missing, output.relative_to(directory).as_posix(),
                    ))
                parent.mkdir(exist_ok=True)
                publication.rename(destination)
            return tuple(exports)
    except Timeout as error:
        raise WebBookOutputError("This rendering is already generating or exporting audio.") from error
    except OSError as error:
        L.w(str(error))
        raise WebBookOutputError("The audiobook audio could not be exported.") from error


def _cut_indices(project: Project, start: int, end: int, lines: tuple[int, ...], use_sections: bool) -> set[int]:
    indices = {_line_index(line, len(project.phrase_groups)) for line in lines}
    if any(not start <= index <= end for index in indices):
        raise WebBookOutputError("Split lines must be within the export range.")
    if use_sections:
        boundaries = {first for first, last in project.book.section_ranges() if first < last}
        if project.chapter_mode is SectionMarkerMode.FILES:
            boundaries.update(project.markers)
        indices.update(index for index in boundaries if start < index <= end)
    indices.discard(start)
    return indices


def _bookmark_indices(
    project: Project, start: int, end: int, lines: tuple[int, ...] | None, cuts: set[int],
) -> list[int]:
    if cuts:
        if lines:
            raise WebBookOutputError("Bookmarks are available for single-file exports. Clear bookmark lines to split into files.")
        return []
    if lines is None:
        if project.chapter_mode is SectionMarkerMode.BOOKMARKS and project.can_use_bookmark_section_markers():
            return sorted(index for index in project.markers if start <= index <= end)
        return []
    indices = sorted({_line_index(line, len(project.phrase_groups)) for line in lines})
    if any(not start <= index <= end for index in indices):
        raise WebBookOutputError("Bookmark lines must be within the export range.")
    return indices


def _make_audio_file(state, first_idx, last_idx, bookmarks, destination, use_upsampler):
    from tts_audiobook_tool.concat_util import ConcatUtil

    if not use_upsampler:
        return ConcatUtil.make_file(state, first_idx, last_idx, bookmarks, destination, use_upsampler=False)

    from tts_audiobook_tool.model_worker import ModelWorker
    from tts_audiobook_tool.model_runtime import mark_interactive_main

    mark_interactive_main()
    try:
        available, error = ModelWorker.probe_lava_sr_blocking()
        if error or not available:
            raise WebBookOutputError("LavaSR v2 is unavailable in the browser server's environment.")
        return ConcatUtil.make_file(state, first_idx, last_idx, bookmarks, destination, use_upsampler=True)
    finally:
        ModelWorker.shutdown()
