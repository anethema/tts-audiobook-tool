from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from typing import cast

from watchdog.events import FileCreatedEvent, FileDeletedEvent

from tts_audiobook_tool.constants import PROJECT_SOUND_SEGMENTS_SUBDIR
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support import project_sound_segments as catalog
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil


def make_project(root: Path, line_count: int = 3) -> Project:
    segments_dir = root / PROJECT_SOUND_SEGMENTS_SUBDIR
    segments_dir.mkdir()
    return cast(Project, SimpleNamespace(
        dir_path=str(root),
        sound_segments_path=str(segments_dir),
        phrase_groups=[SimpleNamespace(text=f"Line {index}") for index in range(line_count)],
    ))


def take_path(
    project: Project, index: int, label: str = "Take", *, suffix: str = ".flac",
    errors: int | None = None, stale: bool = False,
) -> Path:
    text = "Former text" if stale else project.phrase_groups[index].text
    segment_hash = SoundSegmentUtil.calc_segment_hash(index, text)
    score = f" [{errors}]" if errors is not None else ""
    return Path(project.sound_segments_path) / (
        f"[{index + 1:05}] [{segment_hash}] [chatterbox] [voice]{score} {label}{suffix}"
    )


def publish(path: Path, *, sidecar: bool = False) -> None:
    if sidecar:
        path.with_suffix(".json").write_text("{}", encoding="utf-8")
    temporary = path.with_suffix(".temp")
    temporary.write_bytes(b"checkpoint audio")
    temporary.replace(path)


def open_index(project: Project, *, watch: bool = True) -> catalog.ProjectSoundSegments:
    index = catalog.ProjectSoundSegments(project)
    index.dont_show_scan_message = True
    if not watch:
        index.observer.stop()
        index.observer.join(timeout=3)
    return index


def wait_until(predicate) -> None:
    deadline = monotonic() + 5
    while not predicate():
        assert monotonic() < deadline, "checkpoint index did not observe the filesystem change"
        sleep(0.01)


def test_publications_in_large_project_walk_directory_only_on_open_and_recovery(tmp_path, monkeypatch):
    existing_count = 5000
    project = make_project(tmp_path, existing_count + 100)
    for index in range(existing_count):
        take_path(project, index).write_bytes(b"flac")
    unrelated = take_path(project, 0, "Other take", errors=3)
    publish(unrelated)
    old_line = take_path(project, existing_count, stale=True)
    publish(old_line)
    empty = take_path(project, existing_count + 1, suffix=".M4A")
    empty.touch()

    walks = []
    original_iterdir = Path.iterdir

    def track_walk(path):
        if path == Path(project.sound_segments_path):
            walks.append(path)
        return original_iterdir(path)

    ranked = []
    original_rank = catalog.get_segment_error_rank

    def track_rank(item, directory):
        ranked.append(item.idx)
        return original_rank(item, directory)

    monkeypatch.setattr(Path, "iterdir", track_walk)
    monkeypatch.setattr(catalog, "get_segment_error_rank", track_rank)
    checkpoints = open_index(project)
    assert checkpoints.num_generated() == existing_count

    for index in range(existing_count, existing_count + 100):
        failed = take_path(project, index, "Failed", errors=4)
        publish(failed, sidecar=True)
        assert checkpoints.record_published_segment(failed) == 0
        assert checkpoints.get_best_file_for(index) == failed.name

        clean = take_path(project, index, "Clean", suffix=".m4a")
        publish(clean, sidecar=True)
        assert checkpoints.record_published_segment(clean) == 1
        assert checkpoints.get_filenames_for(index) == [clean.name]
        assert not failed.exists() and not failed.with_suffix(".json").exists()

    assert len(walks) == 1
    assert set(ranked) == set(range(existing_count, existing_count + 100))
    assert len(ranked) == 200
    assert unrelated.exists() and old_line.exists() and empty.exists()

    checkpoints.force_invalidate()
    assert checkpoints.num_generated() == existing_count + 100
    assert len(walks) == 2


def test_pruning_verifies_cached_takes_and_preserves_shared_sidecar(tmp_path):
    project = make_project(tmp_path)
    old = take_path(project, 0, "Old")
    publish(old, sidecar=True)
    checkpoints = open_index(project, watch=False)
    assert checkpoints.get_best_file_for(0) == old.name
    old.unlink()  # Deletion has happened on disk while its watcher event is pending.

    replacement = take_path(project, 0, "New", errors=4)
    publish(replacement, sidecar=True)
    assert checkpoints.record_published_segment(replacement) == 0
    assert checkpoints.get_best_file_for(0) == replacement.name

    same_stem = replacement.with_suffix(".m4a")
    publish(same_stem)
    assert checkpoints.record_published_segment(same_stem) == 1
    assert checkpoints.get_filenames_for(0) == [replacement.name]
    assert not same_stem.exists()
    assert replacement.with_suffix(".json").is_file()


def test_published_replacement_takes_effect_only_after_audio_exists(tmp_path):
    project = make_project(tmp_path)
    original = take_path(project, 0, "Old")
    replacement = take_path(project, 0, "Replacement")
    publish(original, sidecar=True)
    checkpoints = open_index(project, watch=False)
    assert checkpoints.record_published_segment(replacement, replaces=original) == 0
    assert checkpoints.get_best_file_for(0) == original.name

    publish(replacement, sidecar=True)
    assert checkpoints.record_published_segment(replacement, replaces=original, prune=False) == 1
    assert checkpoints.get_filenames_for(0) == [replacement.name]
    assert not original.exists() and not original.with_suffix(".json").exists()
    assert replacement.exists() and replacement.with_suffix(".json").exists()

    # Replayed events read current disk state and cannot undo the publication.
    checkpoints.on_dir_contents_change(FileCreatedEvent(str(original)))
    checkpoints.on_dir_contents_change(FileDeletedEvent(str(replacement)))
    assert checkpoints.get_filenames_for(0) == [replacement.name]


def test_audio_delete_failure_preserves_take_and_sidecar(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    take = take_path(project, 0, suffix=".m4a")
    publish(take, sidecar=True)
    checkpoints = open_index(project, watch=False)
    assert checkpoints.get_filenames_for(0) == [take.name]

    def reject_delete(path):
        assert path == take
        raise PermissionError(path)

    with monkeypatch.context() as filesystem:
        filesystem.setattr(Path, "unlink", reject_delete)
        assert not checkpoints.delete_sound_segment_and_sidecars(take)
    assert take.exists() and take.with_suffix(".json").exists()
    assert checkpoints.get_filenames_for(0) == [take.name]

    assert checkpoints.delete_sound_segment_and_sidecars(take)
    assert checkpoints.get_filenames_for(0) == []
    assert not take.with_suffix(".json").exists()


def test_watcher_reconciles_file_changes_and_directory_replacement(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    checkpoints = open_index(project)
    scans = []
    original_scan = SoundSegmentUtil.make_sound_segments_map

    def track_scan(source):
        scans.append(source)
        return original_scan(source)

    monkeypatch.setattr(SoundSegmentUtil, "make_sound_segments_map", track_scan)
    assert checkpoints.get_existing_indices() == set()

    first = take_path(project, 0, suffix=".M4A")
    first.touch()
    first.write_bytes(b"created then filled")
    wait_until(lambda: checkpoints.get_filenames_for(0) == [first.name])
    first.write_bytes(b"")
    wait_until(lambda: checkpoints.get_existing_indices() == set())
    first.write_bytes(b"modified in place")
    wait_until(lambda: checkpoints.get_filenames_for(0) == [first.name])

    renamed = take_path(project, 1, suffix=".flac")
    first.rename(renamed)
    wait_until(lambda: checkpoints.get_existing_indices() == {1})
    outside = tmp_path / renamed.name
    renamed.rename(outside)
    wait_until(lambda: checkpoints.get_existing_indices() == set())
    outside.rename(renamed)
    wait_until(lambda: checkpoints.get_existing_indices() == {1})
    renamed.unlink()
    wait_until(lambda: checkpoints.get_existing_indices() == set())
    publish(renamed)
    wait_until(lambda: checkpoints.get_existing_indices() == {1})
    assert len(scans) == 1

    replacement_directory = tmp_path / "prepared"
    replacement_directory.mkdir()
    replacement = take_path(project, 2, suffix=".m4a")
    publish(replacement_directory / replacement.name)
    segments_directory = Path(project.sound_segments_path)
    segments_directory.rename(tmp_path / "previous")
    replacement_directory.rename(segments_directory)
    wait_until(lambda: checkpoints.get_existing_indices() == {2})
    assert len(scans) >= 2


def test_watcher_change_during_rebuild_is_applied_to_completed_snapshot(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    checkpoints = open_index(project, watch=False)
    scan_ready = Event()
    finish_scan = Event()
    callback_ready = Event()
    original_scan = SoundSegmentUtil.make_sound_segments_map

    def pause_scan(source):
        snapshot = original_scan(source)
        scan_ready.set()
        assert finish_scan.wait(timeout=5)
        return snapshot

    monkeypatch.setattr(SoundSegmentUtil, "make_sound_segments_map", pause_scan)
    take = take_path(project, 0)

    def notify_created():
        callback_ready.set()
        checkpoints.on_dir_contents_change(FileCreatedEvent(str(take)))

    with ThreadPoolExecutor(max_workers=2) as executor:
        reading = executor.submit(checkpoints.get_existing_indices)
        try:
            assert scan_ready.wait(timeout=5)
            publish(take)
            notification = executor.submit(notify_created)
            assert callback_ready.wait(timeout=5)
        finally:
            finish_scan.set()
        assert reading.result(timeout=5) == set()
        notification.result(timeout=5)
    assert checkpoints.get_filenames_for(0) == [take.name]
