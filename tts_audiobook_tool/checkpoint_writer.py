from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from tts_audiobook_tool.app_types import Sound, SoundSegment
from tts_audiobook_tool.app_types.segment_transcript_data import SegmentTranscriptData
from tts_audiobook_tool.project_support.segment_staging_util import list_staged_segments
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.project_support.sound_segment_util import (
    SOUND_SEGMENT_SUFFIXES,
    SoundSegmentUtil,
    get_segment_error_rank,
    get_segment_stt_info_path,
)
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.util import delete_silently, make_error_string


class CheckpointWriteError(RuntimeError):
    pass


@dataclass(frozen=True)
class SavedCheckpoint:
    index: int
    path: str
    error: str
    elapsed: float
    duration: float | None
    passed: bool
    final_attempt: bool
    improved: bool
    quiet: bool


class CheckpointWriteQueue:
    """A single ordered CPU writer with bounded outstanding waveforms."""

    def __init__(self, batch_size: int) -> None:
        self._limit = min(2 * batch_size, 128)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-writer")
        self._pending: deque[Future[SavedCheckpoint]] = deque()

    def __enter__(self) -> CheckpointWriteQueue:
        return self

    def __exit__(self, *_exc) -> None:
        self._executor.shutdown(wait=True)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def submit(self, write: Callable[[], SavedCheckpoint]) -> None:
        if self.pending >= self._limit:
            raise RuntimeError("Checkpoint write queue is full")
        self._pending.append(self._executor.submit(write))

    def collect(self, *, drain: bool = False, make_space: bool = False) -> Iterator[SavedCheckpoint]:
        while self._pending and (drain or self._pending[0].done() or (make_space and self.pending >= self._limit)):
            yield self._pending.popleft().result()


class StagedCheckpoints:
    """Rank and prune staged takes by line without rescanning the staging directory."""

    def __init__(self, directory: str) -> None:
        self._by_index: dict[int, list[tuple[Path, SoundSegment]]] = {}
        for path, segment in list_staged_segments(directory):
            self._by_index.setdefault(segment.idx, []).append((path, segment))

    def record(self, path: Path, replaced_path: Path | None) -> None:
        segment = SoundSegmentUtil.make_from_file_name(path.name)
        if segment is None:
            return
        takes = self._by_index.setdefault(segment.idx, [])
        if replaced_path is not None:
            _remove_take(replaced_path)
            takes[:] = [take for take in takes if take[0] != replaced_path]
        takes.append((path, segment))
        best = min(takes, key=lambda take: (get_segment_error_rank(take[1], path.parent), take[0].name))
        for take in takes:
            if take != best:
                _remove_take(take[0])
        takes[:] = [best]


def choose_checkpoint_path(path: Path) -> tuple[Path, Path | None]:
    """Choose a separate stem so publication can preserve the previous pair."""
    alternate = path.with_stem(path.stem + "~2")
    replaced = next((candidate for candidate in (path, alternate) if candidate.is_file()), None)
    for candidate in (path, alternate):
        collisions = [candidate.with_suffix(suffix) for suffix in SOUND_SEGMENT_SUFFIXES]
        collisions.append(get_segment_stt_info_path(candidate))
        if not any(os.path.lexists(collision) for collision in collisions):
            return candidate, replaced
    return path.with_stem(f"{path.stem}~{uuid4().hex[:12]}"), replaced


def write_checkpoint(
    path: Path,
    sound: Sound,
    transcript: SegmentTranscriptData | None,
    aac_bitrate: str,
) -> str:
    """Write a new take and publish the optional JSON before its audio name."""
    suffix = path.suffix
    temporary_stem = path.parent / f".checkpoint-{uuid4().hex}"
    temp_sound = temporary_stem.with_suffix(suffix)
    temp_sidecar = temporary_stem.with_suffix(".json")
    sidecar = get_segment_stt_info_path(path)
    published_audio = published_sidecar = False
    try:
        if suffix == ".m4a":
            error = SoundFileUtil.save_aac(sound, str(temp_sound), aac_bitrate)
        else:
            error = SoundFileUtil.save_flac(sound, str(temp_sound))
        if error:
            return error
        _sync_file(temp_sound)
        if transcript is not None:
            error = SegmentTranscriptUtil.save(temp_sidecar, transcript)
            if error:
                return error
            _sync_file(temp_sidecar)
            os.replace(temp_sidecar, sidecar)
            published_sidecar = True
            _sync_directory(path.parent)
        os.replace(temp_sound, path)
        published_audio = True
        _sync_directory(path.parent)
        return ""
    except OSError as error:
        if published_audio:
            delete_silently(str(path))
        if published_sidecar:
            delete_silently(str(sidecar))
        return make_error_string(error)
    finally:
        delete_silently(str(temp_sound))
        delete_silently(str(temp_sidecar))


def _sync_file(path: Path) -> None:
    with path.open("r+b") as file:
        os.fsync(file.fileno())


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _remove_take(path: Path) -> None:
    path.unlink(missing_ok=True)
    if not any(path.with_suffix(suffix).exists() for suffix in SOUND_SEGMENT_SUFFIXES):
        get_segment_stt_info_path(path).unlink(missing_ok=True)
