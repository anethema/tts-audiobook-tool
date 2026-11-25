import os
from pathlib import Path
from threading import Lock
from typing import Callable, Collection, Iterator
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from tts_audiobook_tool.project_support.sound_segment_util import (
    SOUND_SEGMENT_SUFFIXES,
    SoundSegment,
    SoundSegmentUtil,
    get_segment_error_rank,
    get_segment_stt_info_path,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.constants import COL_DIM_ITALICS
from tts_audiobook_tool.system_support.ansi import Ansi
from tts_audiobook_tool.util import delete_silently, printt
from tts_audiobook_tool.sound.checkpoint_audio import has_complete_audio
from tts_audiobook_tool.validator import Validator
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings

class ProjectSoundSegments:
    """Index the project's saved takes by text line and reconcile file changes."""

    def __init__(self, project: Project):
        self.project = project # TODO: circular reference, ng
        self._lock = Lock()
        self._sound_segments_map: dict[int, list[SoundSegment]] = {}
        self._dirty = True
        self.dont_show_scan_message = False
        self._segments_dir: Path | None = None

        event_handler = DirHandler(self.on_dir_contents_change)
        self.observer = Observer()

        if project.dir_path:
            self._segments_dir = Path(os.path.abspath(project.sound_segments_path))
            # Watch the project directory (parent) instead of segments/ directly.
            # This is more reliable: watchdog can miss self-deletion events for the
            # watched directory itself, but child events (segments dir being deleted,
            # files created/deleted within segments/) are reliably delivered.
            self.observer.schedule(event_handler, os.path.abspath(project.dir_path), recursive=True)
            self.observer.start()

    def force_invalidate(self) -> None:
        """Rebuild from disk on the next lookup, including changes missed by the watcher."""
        with self._lock:
            self._dirty = True

    def on_dir_contents_change(self, event: FileSystemEvent) -> None:
        if self._segments_dir is None or (event.is_directory and event.event_type == 'modified'):
            return
        paths = tuple(
            Path(os.path.abspath(os.fsdecode(path)))
            for path in (event.src_path, event.dest_path) if path
        )
        if event.is_directory:
            if event.event_type in ('created', 'deleted', 'moved') and self._segments_dir in paths:
                self.force_invalidate()
            return

        segment_paths = [
            path for path in paths
            if path.parent == self._segments_dir and path.suffix.lower() in SOUND_SEGMENT_SUFFIXES
        ]
        if not segment_paths:
            return
        with self._lock:
            if self._dirty:
                return
            try:
                for path in segment_paths:
                    # The disk may be ahead of the event queue. Reconcile both sides
                    # of a move using the current state, including delayed self events.
                    self._sync_file(path)
            except OSError:
                self._dirty = True

    def _sync_file(self, path: Path) -> SoundSegment | None:
        """Reconcile a single supported path while holding the index lock."""
        if path.parent != self._segments_dir or path.suffix.lower() not in SOUND_SEGMENT_SUFFIXES:
            return None
        segment = SoundSegmentUtil.make_from_file_name(path.name)
        if segment is None:
            return None
        is_current = SoundSegmentUtil.is_current_segment(segment, self.project) and (
            SoundSegmentUtil.is_nonempty_file(path)
        )
        items = self._sound_segments_map.get(segment.idx, [])
        name = os.path.normcase(segment.file_name)
        for position, item in enumerate(items):
            if os.path.normcase(item.file_name) != name:
                continue
            if is_current:
                items[position] = segment
            else:
                del items[position]
                if not items:
                    del self._sound_segments_map[segment.idx]
            break
        else:
            if is_current:
                items.append(segment)
                self._sound_segments_map[segment.idx] = items
        return segment if is_current else None

    def _rebuild_if_needed(self) -> None:
        if not self._dirty:
            return
        show_scan_message = not self.dont_show_scan_message
        if show_scan_message:
            printt(f"{COL_DIM_ITALICS}Project directory contents have changed. Scanning...", end="\r")
        try:
            self._sound_segments_map = SoundSegmentUtil.make_sound_segments_map(self.project)
            self._dirty = False
        finally:
            if show_scan_message:
                printt(f"{Ansi.ERASE_REST_OF_LINE}", end="\r")

    @property
    def sound_segments_map(self) -> dict[int, list[SoundSegment]]:
        with self._lock:
            self._rebuild_if_needed()
            return {index: items.copy() for index, items in self._sound_segments_map.items()}

    def get_items_for(self, index: int) -> list[SoundSegment]:
        with self._lock:
            self._rebuild_if_needed()
            return self._sound_segments_map.get(index, []).copy()

    def get_filenames_for(self, index: int) -> list[str]:
        with self._lock:
            self._rebuild_if_needed()
            return [item.file_name for item in self._sound_segments_map.get(index, [])]

    def get_best_item_for(self, index: int) -> SoundSegment | None:
        """Return the take with the lowest word error rank for the given line."""
        with self._lock:
            self._rebuild_if_needed()
            return self._best_item(self._sound_segments_map.get(index, []))

    def _best_item(self, items: list[SoundSegment]) -> SoundSegment | None:
        if len(items) <= 1:
            # The common case; ranking would only cost a sidecar lookup.
            return items[0] if items else None
        # min() keeps the first of equally ranked takes.
        return min(
            items,
            key=lambda item: get_segment_error_rank(item, self.project.sound_segments_path),
        )

    def get_best_file_for(self, index: int) -> str:
        item = self.get_best_item_for(index)
        return item.file_name if item else ""

    def get_export_file_for(self, index: int) -> str:
        """Select the highest-ranked intact alternate if an interrupted take exists."""
        items = self.get_items_for(index)
        if len(items) <= 1:
            # The exporter will decode the only take. Avoid opening it twice.
            return items[0].file_name if items else ""
        directory = Path(self.project.sound_segments_path)
        for item in sorted(items, key=lambda take: get_segment_error_rank(take, directory)):
            if has_complete_audio(directory / item.file_name):
                return item.file_name
        raise OSError(f"None of the saved takes for line {index + 1} is readable.")

    # ---

    def num_generated(self) -> int:
        with self._lock:
            self._rebuild_if_needed()
            return len(self._sound_segments_map)

    def num_generated_in_current_range(self) -> int:
        indices = ProjectUtil.get_indices_to_generate(self.project)
        with self._lock:
            self._rebuild_if_needed()
            return len(self._sound_segments_map.keys() & indices)

    def get_existing_indices(self) -> set[int]:
        """
        Returns the set of indices of existing sound segments
        """
        with self._lock:
            self._rebuild_if_needed()
            return set(self._sound_segments_map)

    def is_segment_failed(self, index: int, item: SoundSegment) -> bool:
        """
        Determines whether a sound segment should be considered 'failed'
        based on its num_errors and the project's current strictness setting.

        Unknown error count (num_errors == -1) is treated as not-failed.
        """
        if item.num_errors == -1:
            # Unknown error count — treat as not-failed
            return False
        phrase_group = self.project.phrase_groups[index]
        normalized_source = TextNormalizer.normalize_source(phrase_group.text, self.project.language_code)
        num_words = app_text.get_word_count(normalized_source, vocalizable_only=True)
        threshold = Validator.compute_threshold(num_words, self.project.strictness)
        return ValidationFindings.is_legacy_filename_score_failed(item.num_errors, threshold)

    def get_failed_indices_in_generate_range(self) -> set[int]:
        """
        Within the project's defined 'generate range',
        returns the indices of sound segments that exist but are all failed.

        Uses dynamic failure detection based on num_errors and project strictness.
        """
        all_indices = ProjectUtil.get_indices_to_generate(self.project)
        sound_segments_map = self.sound_segments_map
        failed_indices = set()
        for index in all_indices:
            items = sound_segments_map.get(index, [])
            if not items:
                continue
            for item in items:
                if self.is_segment_failed(index, item):
                    failed_indices.add(index)
                    break
        return failed_indices

    def get_word_error_counts_in_generate_range(self) -> dict[int, int]:
        """
        Within the project's defined 'generate range',
        returns the fail counts of existing sound segment items
        """
        all_indices = ProjectUtil.get_indices_to_generate(self.project)
        sound_segments_map = self.sound_segments_map
        fail_counts = {}
        for index in all_indices:
            items = sound_segments_map.get(index, [])
            if not items:
                continue
            for item in items:
                fail_counts[index] = item.num_errors
        return fail_counts

    def delete_all(self) -> None:
        with self._lock:
            self._rebuild_if_needed()
            self._delete_indices(tuple(self._sound_segments_map))

    def delete_sound_segment_and_sidecars(self, sound_file_path: Path) -> bool:
        """Delete a take and its timing sidecar. Return whether audio was removed."""
        with self._lock:
            return self._delete_file(Path(os.path.abspath(sound_file_path)))

    def _delete_file(self, path: Path) -> bool:
        try:
            path.unlink()
        except (FileNotFoundError, NotADirectoryError):
            removed = False
        except OSError:
            return False
        else:
            removed = True

        segment = SoundSegmentUtil.make_from_file_name(path.name)
        if not self._dirty:
            self._sync_file(path)
        sidecar = get_segment_stt_info_path(path)
        # Two audio extensions can share a stem in a mixed-format project.
        # Keep the timing sidecar when another take still uses that stem.
        alternate_paths = {path.with_suffix(suffix) for suffix in SOUND_SEGMENT_SUFFIXES}
        if segment is not None and path.parent == self._segments_dir:
            alternate_paths.update(
                path.parent / item.file_name
                for item in self._sound_segments_map.get(segment.idx, [])
                if get_segment_stt_info_path(item.file_name) == Path(sidecar.name)
            )
        alternate_paths.discard(path)
        try:
            shared_sidecar = any(alternate.is_file() for alternate in alternate_paths)
        except OSError:
            return removed
        if not shared_sidecar:
            delete_silently(str(sidecar))
        return removed

    def record_published_segment(
        self,
        sound_file_path: str | Path,
        *,
        prune: bool = True,
        replaces: str | Path | None = None,
    ) -> int:
        """Index a published take, then optionally keep only its line's best take.

        Call after publishing the audio and any timing sidecar. ``replaces``
        identifies an earlier same-format take superseded by this publication.
        Return the number of takes actually removed.
        """
        path = Path(os.path.abspath(sound_file_path))
        if path.parent != self._segments_dir or path.suffix.lower() not in SOUND_SEGMENT_SUFFIXES:
            return 0
        with self._lock:
            self._rebuild_if_needed()
            segment = self._sync_file(path)
            if segment is None:
                return 0
            removed = 0
            if replaces is not None:
                previous = Path(os.path.abspath(replaces))
                if previous != path and previous.suffix.lower() == path.suffix.lower():
                    old_segment = SoundSegmentUtil.make_from_file_name(previous.name)
                    if (
                        previous.parent == path.parent
                        and old_segment is not None
                        and old_segment.idx == segment.idx
                        and old_segment.hash == segment.hash
                    ):
                        removed += self._delete_file(previous)
            if prune:
                removed += self._prune_line(segment.idx)
            return removed

    def delete_redundants_for(self, index: int) -> int:
        """Keep the line's best take and return the number of audio files removed."""
        with self._lock:
            self._rebuild_if_needed()
            return self._prune_line(index)

    def _prune_line(self, index: int) -> int:
        items = self._sound_segments_map.get(index, [])
        if len(items) <= 1 or self._segments_dir is None:
            return 0
        # An external edit may still be in the watcher queue. Check these
        # takes before choosing the survivor or deleting any newly saved audio.
        for item in items.copy():
            self._sync_file(self._segments_dir / item.file_name)
        best_item = self._best_item(items)
        return sum(
            self._delete_file(self._segments_dir / item.file_name)
            for item in items.copy() if item != best_item
        )

    def delete_by_indices(self, indices: Collection[int]) -> None:
        """Delete present takes at the selected line indices."""
        with self._lock:
            self._rebuild_if_needed()
            self._delete_indices(indices)

    def _delete_indices(self, indices: Collection[int]) -> None:
        if self._segments_dir is None:
            return
        for index in indices:
            for item in self._sound_segments_map.get(index, []).copy():
                self._delete_file(self._segments_dir / item.file_name)

    def snapshot_paths_from_index(self, first_index: int) -> list[Path]:
        """Snapshot recognizable generated sound paths at or after an index."""
        if first_index < 0:
            return []
        return [path for path, segment in self._recognizable_files() if segment.idx >= first_index]

    def snapshot_paths_at_indices(self, indices: Collection[int]) -> list[Path]:
        """Snapshot recognizable generated sound paths whose index is in ``indices``."""
        wanted = set(indices)
        if not wanted:
            return []
        return [path for path, segment in self._recognizable_files() if segment.idx in wanted]

    def _recognizable_files(self) -> Iterator[tuple[Path, SoundSegment]]:
        if self._segments_dir is None:
            return
        try:
            for path in self._segments_dir.iterdir():
                if path.suffix.lower() not in SOUND_SEGMENT_SUFFIXES or not path.is_file():
                    continue
                segment = SoundSegmentUtil.make_from_file_name(path.name)
                if segment is not None:
                    yield path, segment
        except (FileNotFoundError, NotADirectoryError):
            return

    def delete_path_snapshot(self, paths: Collection[Path]) -> None:
        """Delete a pre-mutation path snapshot and invalidate the cached catalog."""
        with self._lock:
            try:
                for path in paths:
                    self._delete_file(Path(os.path.abspath(path)))
            finally:
                self._dirty = True

# ---

class DirHandler(FileSystemEventHandler):

    def __init__(self, callback: Callable[[FileSystemEvent], None]):
        self.callback = callback

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type in ('created', 'deleted', 'moved', 'modified', 'closed'):
            self.callback(event)
