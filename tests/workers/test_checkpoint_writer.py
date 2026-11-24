from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest

from generate_files_test_support import generate_files_mock_stack
from tts_audiobook_tool import checkpoint_writer
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types import Book, BookSection, Sound, SttVariant
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings
from tts_audiobook_tool.app_types.validation_result import SkippedResult, WordErrorResult
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationPhase, GenerationProgress, GenerationRunEnded
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.project_support.segment_staging_util import list_staged_segments
from tts_audiobook_tool.project_support.sound_segment_util import get_segment_stt_info_path
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State


@pytest.fixture
def aac_state(tmp_path):
    groups = [PhraseGroup([Phrase(f"Line number {i}.", Reason.SENTENCE)]) for i in range(4)]
    project = Project(
        dir_path=str(tmp_path), book=Book([BookSection(phrase_groups=groups)]),
        checkpoint_format="aac", max_retries=0,
    )
    # Use real preferences and project data, with synthesis replaced per test.
    state = State.for_worker(Prefs(project_dir=str(tmp_path), stt_variant=SttVariant.DISABLED, aac_bitrate="64k"))
    state._project = project
    try:
        yield state
    finally:
        project.kill()


@pytest.fixture
def clip():
    time = np.arange(12_231, dtype=np.float32) / 24_000
    return Sound(0.08 * np.sin(2 * np.pi * time * 440), 24_000)


def skipped(clip):
    result = SkippedResult(clip, "Validation disabled")
    result.voice_tag = "voice"
    return result


def transcript(clip, *, passed: bool = True):
    findings = ValidationFindings() if passed else ValidationFindings(["wrong word"])
    result = WordErrorResult(clip, [], num_words=3, threshold=0, findings=findings)
    result.voice_tag = "voice"
    return result


@pytest.mark.no_ffmpeg_exe
def test_aac_writer_overlaps_batches_drains_on_stop_and_resumes(aac_state, clip, monkeypatch):
    first_write_started = Event()
    second_batch_started = Event()
    writer_can_finish = Event()
    calls = []
    written = []
    progress = []
    real_save = SoundFileUtil.save_aac

    def generate(indices, **_):
        calls.append(tuple(indices))
        if len(calls) == 2:
            assert first_write_started.wait(10)
            assert not written
            Interrupts().signal_handler(None, None)
            second_batch_started.set()
        return [skipped(clip) for _ in indices]

    def save(sound, path, bitrate):
        assert bitrate == "64k"
        if not written:
            first_write_started.set()
            assert second_batch_started.wait(10)
        if len(written) == 1:
            assert writer_can_finish.wait(10)
        error = real_save(sound, path, bitrate)
        written.append(path)
        return error

    def track(update):
        if isinstance(update, GenerationPhase) and update.label == "Saving audio":
            assert len(written) <= 1
            writer_can_finish.set()
        if isinstance(update, GenerationProgress):
            assert update.saved <= len(written)
            progress.append(update.saved)
        if isinstance(update, GenerationRunEnded):
            assert len(written) == 2

    monkeypatch.setattr(SoundFileUtil, "save_aac", save)
    project = aac_state.project
    with generate_files_mock_stack(generate, mock_save=False), GenerationEvents.using_sink(track):
        assert GenerateUtil.generate_files(aac_state, {0, 1, 2}, batch_size=1, is_regen=False)

    assert calls == [(0,), (1,)]
    assert progress[-1] == 2
    assert project.sound_segments.get_existing_indices() == {0, 1}
    assert ProjectUtil.get_indices_to_generate(project) == {2, 3}
    project.sound_segments.force_invalidate()
    assert project.sound_segments.get_existing_indices() == {0, 1}

    monkeypatch.setattr(SoundFileUtil, "save_aac", real_save)
    with generate_files_mock_stack(generate, mock_save=False):
        assert not GenerateUtil.generate_files(aac_state, {2}, batch_size=1, is_regen=False)
    assert calls[-1] == (2,)
    assert ProjectUtil.get_indices_to_generate(project) == {3}
    assert len(list(Path(project.sound_segments_path).glob("*.m4a"))) == 3


@pytest.mark.no_ffmpeg_exe
def test_aac_writer_propagates_sidecar_failure_without_saving_failed_line(aac_state, clip, monkeypatch):
    progress = []
    monkeypatch.setattr(checkpoint_writer.SegmentTranscriptUtil, "save", lambda *_: "Drive is full")

    def generate(indices, **_):
        return [skipped(clip) if index == 0 else transcript(clip) for index in indices]

    with (
        generate_files_mock_stack(generate, mock_save=False),
        GenerationEvents.using_sink(progress.append),
        pytest.raises(checkpoint_writer.CheckpointWriteError, match="Drive is full"),
    ):
        GenerateUtil.generate_files(aac_state, {0, 1}, batch_size=2, is_regen=False)

    project = aac_state.project
    assert project.sound_segments.get_existing_indices() == {0}
    assert ProjectUtil.get_indices_to_generate(project) == {1, 2, 3}
    assert [event for event in progress if isinstance(event, GenerationProgress)][-1].saved == 1
    assert not list(Path(project.sound_segments_path).glob(".*checkpoint*"))


@pytest.mark.no_ffmpeg_exe
def test_replacing_a_take_publishes_timing_first_and_preserves_old_pair_on_failure(aac_state, clip, monkeypatch):
    project = aac_state.project
    group = project.phrase_groups[0]
    result = transcript(clip)
    error, path = GenerateUtil.save_sound_and_timing_json(aac_state, group, 0, result, is_real_time=False, voice_tag="voice")
    assert not error
    original = Path(path)
    old_timing = get_segment_stt_info_path(original)
    old_bytes = original.read_bytes(), old_timing.read_bytes()
    real_replace = checkpoint_writer.os.replace

    def replace(source, destination):
        destination = Path(destination)
        if destination.suffix == ".m4a" and destination.name.startswith("["):
            assert destination != original
            assert get_segment_stt_info_path(destination).is_file()
            assert original.is_file() and old_timing.is_file()
            raise OSError("Atomic publication failed")
        return real_replace(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(checkpoint_writer.os, "replace", replace)
        error, replacement = GenerateUtil.save_sound_and_timing_json(
            aac_state, group, 0, result, is_real_time=False, voice_tag="voice",
        )

    assert "Atomic publication failed" in error
    assert (original.read_bytes(), old_timing.read_bytes()) == old_bytes
    assert not Path(replacement).exists()
    assert not get_segment_stt_info_path(replacement).exists()
    error, replacement = GenerateUtil.save_sound_and_timing_json(aac_state, group, 0, result, is_real_time=False, voice_tag="voice")
    assert not error
    assert not original.exists() and not old_timing.exists()
    assert project.sound_segments.get_best_file_for(0) == Path(replacement).name
    assert get_segment_stt_info_path(replacement).is_file()


@pytest.mark.no_ffmpeg_exe
def test_aac_staging_ranks_retries_with_one_directory_listing(aac_state, clip, tmp_path):
    directory = str(tmp_path / ".regeneration")
    aac_state.project.max_retries = 1
    results = iter([transcript(clip, passed=False), transcript(clip)])
    with (
        generate_files_mock_stack(lambda **_: [next(results)], mock_save=False),
        patch.object(checkpoint_writer, "list_staged_segments", wraps=list_staged_segments) as listing,
    ):
        assert not GenerateUtil.generate_files(aac_state, {0}, batch_size=1, is_regen=True, staging_dir=directory)
    listing.assert_called_once_with(directory)
    [(audio, take)] = list_staged_segments(directory)
    assert take.num_errors == -1 and audio.suffix == ".m4a" and get_segment_stt_info_path(audio).is_file()
    assert not aac_state.project.sound_segments.get_existing_indices()
