"""Dispatch ordered cast requests and independent delivery requests for book analysis."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TypeVar

from tts_audiobook_tool.speaker_assign.assign_plan import PlannedChunk


@dataclass(frozen=True)
class ChunkAnalysisRequest:
    chunk: PlannedChunk
    total: int
    speakers: set[int] = field(default_factory=set)
    cues: set[int] = field(default_factory=set)
    voices: set[str] = field(default_factory=set)
    delivery_only: bool = False

    @property
    def label(self) -> str:
        return f"Chunk {self.chunk.number:,}/{self.total:,}"

    @property
    def line_range(self) -> str:
        return f"lines {self.chunk.start + 1:,}–{self.chunk.end:,}"

    @property
    def location(self) -> str:
        return f"{self.label} ({self.line_range})"


T = TypeVar("T")


def run_analysis_requests(
        ordered: deque[ChunkAnalysisRequest], ready_cues: deque[ChunkAnalysisRequest], slots: int,
        prepare: Callable[[ChunkAnalysisRequest], Callable[[], T]],
        collect: Callable[[ChunkAnalysisRequest, Callable[[], T]], None],
        stopped: Callable[[], bool],
) -> None:
    """Only the caller builds prompts, applies replies and checkpoints the book."""
    if slots == 1:
        for request in ordered:
            collect(request, prepare(request))
            if stopped():
                break
        return

    # A cast request gets first choice of a free slot. Its successor needs the
    # committed result. Independent cue requests can use every remaining slot.
    with ThreadPoolExecutor(max_workers=slots, thread_name_prefix="book-analysis") as pool:
        active = {}
        cast_running = False
        while True:
            if not stopped():
                if ordered and not cast_running and len(active) < slots:
                    request = ordered.popleft()
                    active[pool.submit(prepare(request))] = request
                    cast_running = True
                while ready_cues and len(active) < slots:
                    request = ready_cues.popleft()
                    active[pool.submit(prepare(request))] = request
            if not active:
                return
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                request = active.pop(future)
                if not request.delivery_only:
                    cast_running = False
                # In-flight successes still get saved when a different request fails.
                collect(request, future.result)
