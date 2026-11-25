"""Session-scoped console output and audiobook progress for the browser."""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from html import escape
import math
import re
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import gradio as gr

from tts_audiobook_tool.util import duration_string, format_eta_duration

if TYPE_CHECKING:
    from tts_audiobook_tool.generation_events import BookRenderTelemetry


CONSOLE_LIMIT = 40_000
ACTIVITY_CSS = """
#console-output textarea { font-family: var(--font-mono); font-size: var(--text-sm); }
.book-progress-details { display: flex; justify-content: space-between; flex-wrap: wrap; gap: .25rem 1rem; }
.book-progress-track { position: relative; margin-top: .25rem; }
.book-progress-track progress { display: block; width: 100%; height: 1.5rem; accent-color: var(--color-accent); }
.book-progress-rate { position: absolute; top: 50%; right: .4rem; transform: translateY(-50%);
                      background: var(--background-fill-primary); color: var(--body-text-color);
                      border-radius: .25rem; padding: 0 .3rem; font-variant-numeric: tabular-nums; }
"""


@dataclass(frozen=True)
class BookActivity:
    label: str
    telemetry: BookRenderTelemetry | None = None
    ended: bool = False

    def html(self) -> str:
        progress = self.telemetry
        if progress is not None and progress.total > 0:
            done = min(max(0, progress.processed), progress.total)
            details = f"{done:,} of {progress.total:,} segments · {done / progress.total:.0%}"
            bar_values = f'max="{progress.total}" value="{done}"'
        else:
            details = ""
            bar_values = 'max="1" value="0"' if self.ended else ""
        if not self.ended and (
            progress is None
            or (not progress.assembling and (progress.total <= 0 or progress.processed < progress.total))
        ):
            eta = progress.eta_seconds if progress is not None else None
            duration = format_eta_duration(eta)
            estimate = f'<span title="Estimated time to finish generating audio">ETA {duration}</span>'
            details = f"{details} · {estimate}" if details else estimate
        rate = progress.speed_multiplier if progress is not None else None
        speed = ""
        if rate is not None and math.isfinite(rate) and rate >= 0:
            speed = (
                '<output class="book-progress-rate" aria-label="Synthesis speed" '
                f'title="Audio seconds generated per second of synthesis"><strong>{rate:.1f}x</strong></output>'
            )
        return (
            '<div class="book-progress-details">'
            f'<span>{escape(self.label)}</span><span>{details}</span></div>'
            '<div class="book-progress-track">'
            f'<progress aria-label="Overall audiobook progress" {bar_values}></progress>{speed}</div>'
        )


@dataclass(frozen=True)
class _AnalysisPace:
    completed: int
    total: int
    started: float
    sampled: float

    def eta(self, now: float) -> str:
        if not self.completed or self.sampled <= self.started:
            return "measuring…"
        seconds_per_unit = (self.sampled - self.started) / self.completed
        remaining = seconds_per_unit * (self.total - self.completed) - (now - self.sampled)
        return format_eta_duration(remaining) if remaining > 0 else "recalculating…"


@dataclass(frozen=True)
class AnalysisActivity:
    completed: int
    total: int
    phase: str
    detail: str
    started: float
    pace: _AnalysisPace | None = None

    def html(self) -> str:
        now = monotonic()
        elapsed = duration_string(int(now - self.started))
        estimate = ""
        if not self.total or self.completed < self.total:
            duration = self.pace.eta(now) if self.pace else "measuring…"
            estimate = f' · <span title="Estimated time remaining for book analysis">ETA {duration}</span>'
        counts = f"{self.completed:,} of {self.total:,} chunks complete · " if self.total else ""
        values = f'max="{self.total}" value="{self.completed}"' if self.total else ""
        detail = f"<div>{escape(self.detail)}</div>" if self.detail else ""
        return (
            '<div class="book-progress-details">'
            f'<span>{escape(self.phase)}</span><span>{counts}Elapsed {elapsed}{estimate}</span></div>'
            f'{detail}<div class="book-progress-track">'
            f'<progress aria-label="Overall book analysis progress" {values}></progress></div>'
        )


class SessionActivity:
    def __init__(self, *, terminal_log: bool = False):
        self._lock = Lock()
        self._terminal_log = terminal_log
        self._console = ""
        self._book: BookActivity | None = None
        self._console_version = 0
        self._book_version = 0
        self._analysis: AnalysisActivity | None = None
        self._analysis_version = 0

    def write(self, text: str) -> None:
        if text:
            with self._lock:
                self._console = (self._console + text[-CONSOLE_LIMIT:])[-CONSOLE_LIMIT:]
                self._console_version += 1
            if self._terminal_log:
                with suppress(OSError, UnicodeError):
                    print(text, end="", flush=True)

    def start_book(self) -> None:
        with self._lock:
            self._book = BookActivity("Preparing audiobook")
            self._book_version += 1

    def update_book(self, value: BookRenderTelemetry) -> None:
        if value.assembling:
            label = "Assembling M4B"
        elif value.total > 0 and value.processed >= value.total:
            label = "Finalizing audio"
        else:
            label = "Generating audiobook"
        with self._lock:
            self._book = BookActivity(label, value)
            self._book_version += 1

    def finish_book(self, label: str) -> None:
        with self._lock:
            telemetry = self._book.telemetry if self._book else None
            self._book = BookActivity(label, telemetry, ended=True)
            self._book_version += 1

    def update_analysis(
        self, completed: int, total: int, phase: str, detail: str = "", *, work: tuple[int, int] | None = None,
    ) -> None:
        with self._lock:
            now = monotonic()
            started = self._analysis.started if self._analysis else now
            pace = self._analysis.pace if self._analysis else None
            if work is not None and (pace is None or work != (pace.completed, pace.total)):
                pace = _AnalysisPace(*work, started=pace.started if pace else now, sampled=now)
            self._analysis = AnalysisActivity(completed, total, phase, detail, started, pace)
            self._analysis_version += 1

    def finish_analysis(self) -> None:
        with self._lock:
            self._analysis = None
            self._analysis_version += 1

    def changes_since(self, versions: tuple[int, int, int]):
        with self._lock:
            current = self._console_version, self._book_version, self._analysis_version
            console = self._console if current[0] != versions[0] else gr.skip()
            if self._book is not None and current[1] != versions[1]:
                progress = gr.update(value=self._book.html(), visible=True)
            else:
                progress = gr.skip()
            if self._analysis is not None:
                analysis = gr.update(value=self._analysis.html(), visible=True)
            elif current[2] != versions[2]:
                analysis = gr.update(value="", visible=False)
            else:
                analysis = gr.skip()
        return console, progress, analysis, current if current != versions else gr.skip()


class ActivityPanel:
    def __init__(self, *, terminal_log: bool = False):
        self._lock = Lock()
        self._terminal_log = terminal_log
        self._sessions: dict[str | None, SessionActivity] = {}
        self._browser_ids: dict[str, str] = {}
        # A running job retains its activity even after the last tab disconnects.
        self._browsers: WeakValueDictionary[str, SessionActivity] = WeakValueDictionary()

    def bind_browser(self, session_id: str | None, token: str) -> None:
        if session_id is None or not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
            return
        with self._lock:
            activity = self._browsers.get(token)
            if activity is None and session_id not in self._browser_ids:
                activity = self._sessions.get(session_id)
            if activity is None:
                activity = SessionActivity(terminal_log=self._terminal_log)
            self._browser_ids[session_id] = token
            self._sessions[session_id] = self._browsers[token] = activity

    def browser_for(self, session_id: str | None) -> str | None:
        with self._lock:
            return self._browser_ids.get(session_id) if session_id is not None else None

    def session(self, session_id: str | None) -> SessionActivity:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionActivity(terminal_log=self._terminal_log)
            return self._sessions[session_id]

    def console_for(self, session_id: str | None):
        return self.session(session_id).write

    def disconnect(self, session_id: str | None) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._browser_ids.pop(session_id, None)

    def bind(self, app: gr.Blocks, progress: gr.HTML, analysis_progress: gr.HTML) -> None:
        console = gr.Textbox(
            label="Console output", placeholder="No output yet. Generation activity will appear here.",
            lines=12, max_lines=12, interactive=False, buttons=["copy"],
            elem_id="console-output",
        )
        timer = gr.Timer(0.75)
        versions = gr.State((0, 0, 0))

        def refresh_activity(last_seen, request: gr.Request):
            with self._lock:
                activity = self._sessions.get(request.session_hash)
            if activity is None:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip()
            return activity.changes_since(last_seen)

        def discard_activity(request: gr.Request):
            self.disconnect(request.session_hash)

        timer.tick(
            refresh_activity, versions, [console, progress, analysis_progress, versions],
            queue=False, show_progress="hidden", api_visibility="private",
        )
        app.unload(discard_activity)
