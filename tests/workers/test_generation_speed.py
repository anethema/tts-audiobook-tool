from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from generate_files_test_support import StubValidationResult, generate_files_mock_stack, make_generate_state
from tts_audiobook_tool import generate_util
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationProgress


def test_generation_speed_counts_retained_takes_and_time_spent_retrying(monkeypatch, tmp_path):
    state = make_generate_state(max_retries=2)
    clock = 0.0
    first = tmp_path / "first.flac"
    discarded = tmp_path / "discarded.flac"
    replacement = tmp_path / "replacement.flac"
    attempts = iter([
        (True, 120, 10, first),
        (True, 360, 15, discarded),
        (False, 600, 30, replacement),
        (False, 400, 15, None),
    ])

    def warm_up(_state):
        nonlocal clock
        clock += 5
        return SimpleNamespace(should_stop=False)

    def generate_and_validate(**_):
        nonlocal clock
        failed, duration, elapsed, path = next(attempts)
        clock += elapsed
        result = StubValidationResult(failed)
        result.sound = Sound(np.zeros(duration * 10, dtype=np.float32), 10)
        result.path = path
        return [result]

    def save(_state, _phrase_group, _index, result, **_):
        if result.path is None:
            return "Save failed", ""
        result.path.touch()
        return "", str(result.path)

    def prune(_index):
        discarded.unlink(missing_ok=True)
        if replacement.exists():
            first.unlink(missing_ok=True)

    state.project.sound_segments.delete_redundants_for.side_effect = prune
    monkeypatch.setattr(generate_util, "monotonic", lambda: clock)
    events = []
    with (
        generate_files_mock_stack(generate_and_validate),
        patch.object(generate_util.ModelManager, "warm_up_models", side_effect=warm_up),
        patch.object(generate_util.GenerateUtil, "save_sound_and_timing_json", side_effect=save),
        GenerationEvents.using_sink(events.append),
    ):
        assert not generate_util.GenerateUtil.generate_files(state, {0, 1}, batch_size=1, is_regen=False)

    progress = [event for event in events if isinstance(event, GenerationProgress)]
    assert progress[0].speed_multiplier is None
    completed_batches = [event for event in progress if not event.current_indices]
    assert [event.processed for event in completed_batches] == [0, 0, 1, 2]
    assert [event.speed_multiplier for event in completed_batches] == pytest.approx([8, 4, 10, 8])
