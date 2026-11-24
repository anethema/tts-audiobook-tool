from __future__ import annotations


_RECENT_SAMPLE_WEIGHT = 0.35


class GenerationEta:
    """Estimate remaining generation time from the cost of queued sub-batches."""

    def __init__(self, text_lengths: dict[int, int], batch_size: int) -> None:
        self._text_lengths = text_lengths
        self._batch_size = batch_size
        # Allow for some fixed overhead per call, in the same units as text.
        self._call_overhead = max(1, sum(text_lengths.values()) // (4 * max(1, len(text_lengths))))
        self._remaining_work = 0
        self.remaining_items = 0
        self._retry_tails: dict[int, tuple[int, int]] = {}
        self._last_retry_voice = 0
        self._first_rate: float | None = None
        self._recent_duration = 0.0
        self._recent_work = 0.0

    def _batch_work(self, indices: list[int]) -> int:
        lengths = [self._text_lengths[index] for index in indices]
        # Parallel generation depends on the longest line. Validation and
        # other per-line work also grow with the amount of text in the batch.
        return (self._call_overhead + max(lengths)) * self._batch_size + sum(lengths)

    def add_batch(self, indices: list[int]) -> None:
        self._remaining_work += self._batch_work(indices)
        self.remaining_items += len(indices)

    def start_batch(self, indices: list[int]) -> int:
        work = self._batch_work(indices)
        self._remaining_work -= work
        self.remaining_items -= len(indices)
        return work

    def add_retry(self, index: int, voice: int) -> None:
        # Retry rounds preserve order within each voice. Appending changes
        # only the last chunk of that voice, or starts a new chunk.
        size, longest = self._retry_tails.get(voice, (0, 0))
        if size == self._batch_size:
            size, longest = 0, 0
        length = self._text_lengths[index]
        new_longest = max(longest, length)
        added_work = new_longest - longest
        if size == 0:
            added_work += self._call_overhead
        self._remaining_work += added_work * self._batch_size + length
        self.remaining_items += 1
        self._retry_tails[voice] = size + 1, new_longest
        self._last_retry_voice = max(self._last_retry_voice, voice)

    @property
    def last_pending_retry_batch_size(self) -> int:
        # make_retry_round processes voices in ascending index order.
        return self._retry_tails.get(self._last_retry_voice, (0, 0))[0]

    def start_retry_round(self) -> None:
        self._retry_tails.clear()
        self._last_retry_voice = 0

    def record_duration(self, work: int, duration: float) -> None:
        if duration <= 0.0:
            return
        if self._first_rate is None:
            self._first_rate = duration / work
            return
        # The first call can include lazy initialization. Use it until a
        # later sample is available, then smooth the recent rate by work.
        self._recent_duration += _RECENT_SAMPLE_WEIGHT * (duration - self._recent_duration)
        self._recent_work += _RECENT_SAMPLE_WEIGHT * (work - self._recent_work)

    def seconds_remaining(self) -> float | None:
        if self._remaining_work == 0:
            return 0.0
        if self._recent_work:
            return self._remaining_work * self._recent_duration / self._recent_work
        if self._first_rate is not None:
            return self._remaining_work * self._first_rate
        return None
