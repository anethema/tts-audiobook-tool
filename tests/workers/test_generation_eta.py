"""Exercise generation ETA snapshots with a deterministic clock."""

from collections import deque
from dataclasses import replace
from unittest.mock import patch

import pytest

from generate_files_test_support import StubValidationResult, generate_files_mock_stack, make_generate_state
from tts_audiobook_tool import generate_util
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil, TtsModelError
from tts_audiobook_tool.generation_eta import GenerationEta
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationProgress
from tts_audiobook_tool.state import State


def run_generate_files(
    monkeypatch,
    state: State,
    batch_durations: list[float],
    *,
    batch_size: int = 1,
    batch_results: list[list[StubValidationResult | TtsModelError | str]] | None = None,
    save_seconds: float = 0.0,
) -> tuple[list[GenerationProgress], list[tuple[int, ...]]]:
    now = 1000.0
    monkeypatch.setattr(generate_util, "monotonic", lambda: now)
    calls: list[tuple[int, ...]] = []

    def generate_and_validate_batch(*, indices: list[int], **_: object) -> list:
        nonlocal now
        i = len(calls)
        calls.append(tuple(indices))
        now += batch_durations[i]
        if batch_results is not None:
            assert len(batch_results[i]) == len(indices)
            return batch_results[i]
        return [StubValidationResult(False) for _ in indices]

    def save_and_prune(*_, **__):
        nonlocal now
        now += save_seconds
        return "", "saved.wav"

    events = []
    with (
        generate_files_mock_stack(generate_and_validate_batch, mock_save=False),
        patch.object(GenerateUtil, "save_sound_and_timing_json", side_effect=save_and_prune),
        GenerationEvents.using_sink(events.append),
    ):
        did_interrupt = GenerateUtil.generate_files(
            state, set(range(len(state.project.phrase_groups))), batch_size=batch_size, is_regen=False,
        )

    assert not did_interrupt
    assert len(calls) == len(batch_durations)
    return [event for event in events if isinstance(event, GenerationProgress)], calls


def test_first_batch_publishes_eta_and_later_samples_drop_startup_cost(monkeypatch) -> None:
    state = make_generate_state(num_groups=4, max_retries=0)
    emits, _ = run_generate_files(monkeypatch, state, [300.0, 10.0, 10.0, 10.0])
    pre = [event for event in emits if event.current_indices]
    post = [event for event in emits if not event.current_indices]

    assert pre[0].eta_seconds is None
    assert [event.eta_seconds for event in post] == pytest.approx([900.0, 20.0, 10.0, 0.0])
    assert [event.eta_seconds for event in pre[1:]] == pytest.approx([900.0, 20.0, 10.0])
    assert [event.processed for event in post] == [1, 2, 3, 4]


def test_eta_smooths_changes_then_adapts_to_recent_rate(monkeypatch) -> None:
    state = make_generate_state(num_groups=8, max_retries=0)
    emits, _ = run_generate_files(monkeypatch, state, [10.0] * 2 + [20.0] * 6)
    post = [event for event in emits if not event.current_indices]
    predicted_per_batch = [event.eta_seconds / event.remaining for event in post[1:-1]]

    assert predicted_per_batch[0] == pytest.approx(10.0)
    assert 10.0 < predicted_per_batch[1] < 20.0
    assert predicted_per_batch == sorted(predicted_per_batch)
    assert predicted_per_batch[-1] > 18.0


def test_eta_accounts_for_length_sorted_batches_and_partial_tail(monkeypatch) -> None:
    state = make_generate_state(num_groups=7, max_retries=0)
    texts = ["short", "a few words", "a few words", "a few words", "long sentence " * 30]
    texts.extend([texts[-1]] * 2)
    state.project.phrase_groups = [PhraseGroup([Phrase(text, Reason.SENTENCE)]) for text in texts]

    emits, calls = run_generate_files(monkeypatch, state, [40.0, 10.0, 5.0], batch_size=3)
    post = [event for event in emits if not event.current_indices]

    assert calls == [(4, 5, 6), (1, 2, 3), (0,)]
    assert post[0].eta_seconds < 40.0
    assert 0.0 < post[1].eta_seconds < 10.0
    assert post[2].eta_seconds == 0.0


def test_eta_uses_text_length_for_cjk_even_without_word_bucketing(monkeypatch) -> None:
    state = make_generate_state(num_groups=2, max_retries=0)
    state.project.language_code = "zh"
    state.project.phrase_groups = [
        PhraseGroup([Phrase("你好" * repeats, Reason.SENTENCE)]) for repeats in (30, 1)
    ]

    emits, calls = run_generate_files(monkeypatch, state, [30.0, 3.0])
    post = [event for event in emits if not event.current_indices]

    assert calls == [(0,), (1,)]
    assert 0.0 < post[0].eta_seconds < 10.0


def test_eta_includes_synchronous_checkpoint_writes(monkeypatch) -> None:
    state = make_generate_state(num_groups=3, max_retries=0)
    emits, _ = run_generate_files(monkeypatch, state, [10.0] * 3, save_seconds=3.0)
    post = [event for event in emits if not event.current_indices]

    assert [event.eta_seconds for event in post] == pytest.approx([26.0, 13.0, 0.0])


def test_error_batches_do_not_set_or_skew_generation_rate(monkeypatch) -> None:
    state = make_generate_state(num_groups=5, max_retries=0)
    emits, _ = run_generate_files(
        monkeypatch, state, [999.0, 10.0, 999.0, 10.0, 10.0],
        batch_results=[
            [TtsModelError("model error")],
            [StubValidationResult(False)],
            ["validation error"],
            [StubValidationResult(False)],
            [StubValidationResult(False)],
        ],
    )
    post = [event for event in emits if not event.current_indices]

    assert post[0].eta_seconds is None
    assert [event.eta_seconds for event in post[1:]] == pytest.approx([30.0, 20.0, 10.0, 0.0])
    assert post[-1].errored == 2


def test_eta_counts_pending_retries_before_the_rest_of_the_queue(monkeypatch) -> None:
    state = make_generate_state(num_groups=2, max_retries=1)
    emits, calls = run_generate_files(
        monkeypatch, state, [10.0] * 3,
        batch_results=[[StubValidationResult(failed)] for failed in (True, False, False)],
    )
    post = [event for event in emits if not event.current_indices]

    assert calls == [(0,), (0,), (1,)]
    assert [event.eta_seconds for event in post] == pytest.approx([20.0, 10.0, 0.0])
    assert [event.processed for event in post] == [0, 1, 2]
    assert post[-1].retries == 1


def test_eta_counts_separate_retry_sub_batches_by_voice(monkeypatch) -> None:
    state = make_generate_state(num_groups=4, max_retries=1)
    for index, group in enumerate(state.project.phrase_groups):
        group.voice_index = index % 2
    monkeypatch.setattr(generate_util.ProjectVoiceUtil, "get_voice_values", lambda *_: ["A", "B"])

    emits, calls = run_generate_files(
        monkeypatch, state, [10.0] * 4, batch_size=4,
        batch_results=[[StubValidationResult(failed)] * 2 for failed in (True, True, False, False)],
    )
    post = [event for event in emits if not event.current_indices]

    assert calls == [(0, 2), (1, 3), (0, 2), (1, 3)]
    # All four retries fit within the nominal batch size, but use two voices.
    assert [event.eta_seconds for event in post] == pytest.approx([20.0, 20.0, 10.0, 0.0])
    assert post[-1].retries == 4


def test_incremental_retry_work_matches_the_scheduled_voice_chunks() -> None:
    lengths = {index: length for index, length in enumerate((10, 20, 4, 30, 5, 40, 2))}
    planned = GenerationEta(lengths, batch_size=2)
    retries = GenerationEta(lengths, batch_size=2)
    batches = [[0, 2], [4], [1, 3], [5, 6]]
    for batch in batches:
        planned.add_batch(batch)
    for index in (0, 1, 2, 3, 5, 6, 4):
        retries.add_retry(index, int(index in {1, 3, 5, 6}))

    planned.record_duration(work=1, duration=1.0)
    retries.record_duration(work=1, duration=1.0)
    assert retries.seconds_remaining() == planned.seconds_remaining()
    assert retries.last_pending_retry_batch_size == 2
    assert retries.remaining_items == planned.remaining_items == 7

    retries.start_retry_round()
    assert retries.last_pending_retry_batch_size == 0
    for batch in batches:
        assert retries.start_batch(batch) == planned.start_batch(batch)
    assert retries.seconds_remaining() == 0.0
    assert retries.remaining_items == 0


def test_eta_accounts_for_async_checkpoint_backlog_and_final_drain(monkeypatch) -> None:
    class ScriptedWriter:
        def __init__(self, _batch_size: int) -> None:
            self._pending = deque()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            assert not self._pending

        @property
        def pending(self):
            return len(self._pending)

        def submit(self, write):
            self._pending.append(write)

        def collect(self, *, drain=False, make_space=False):
            while self.pending > (0 if drain else 1):
                yield replace(self._pending.popleft()(), elapsed=20.0)

    state = make_generate_state(num_groups=3, max_retries=0)
    state.project.checkpoint_format = "aac"
    monkeypatch.setattr(generate_util, "CheckpointWriteQueue", ScriptedWriter)

    emits, _ = run_generate_files(monkeypatch, state, [10.0] * 3)
    post = [event for event in emits if not event.current_indices]

    assert [event.eta_seconds for event in post] == pytest.approx([20.0, 40.0, 20.0, 20.0, 0.0])
    assert [event.saved for event in post] == [0, 1, 2, 2, 3]
