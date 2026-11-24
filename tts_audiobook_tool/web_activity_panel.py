"""Session-scoped console output and audiobook progress for the browser."""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from html import escape
import math
from threading import Lock
from typing import TYPE_CHECKING

import gradio as gr

from tts_audiobook_tool.util import duration_string

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
            duration = "measuring…"
            if eta is not None and math.isfinite(eta) and eta > 0:
                duration = duration_string(math.ceil(eta), pad_seconds=True).replace("h", "h ").replace("m", "m ")
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


class SessionActivity:
    def __init__(self, *, terminal_log: bool = False):
        self._lock = Lock()
        self._terminal_log = terminal_log
        self._console = ""
        self._book: BookActivity | None = None
        self._console_version = 0
        self._book_version = 0

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

    def changes_since(self, versions: tuple[int, int]):
        with self._lock:
            current = self._console_version, self._book_version
            console = self._console if current[0] != versions[0] else gr.skip()
            if self._book is not None and current[1] != versions[1]:
                progress = gr.update(value=self._book.html(), visible=True)
            else:
                progress = gr.skip()
        return console, progress, current if current != versions else gr.skip()


class ActivityPanel:
    def __init__(self, *, terminal_log: bool = False):
        self._lock = Lock()
        self._terminal_log = terminal_log
        self._sessions: dict[str | None, SessionActivity] = {}

    def session(self, session_id: str | None) -> SessionActivity:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionActivity(terminal_log=self._terminal_log)
            return self._sessions[session_id]

    def console_for(self, session_id: str | None):
        return self.session(session_id).write

    def bind(self, app: gr.Blocks, progress: gr.HTML) -> None:
        console = gr.Textbox(
            label="Console output", placeholder="No output yet. Generation activity will appear here.",
            lines=12, max_lines=12, interactive=False, buttons=["copy"],
            elem_id="console-output",
        )
        timer = gr.Timer(0.75)
        versions = gr.State((0, 0))

        def refresh_activity(last_seen, request: gr.Request):
            with self._lock:
                activity = self._sessions.get(request.session_hash)
            if activity is None:
                return gr.skip(), gr.skip(), gr.skip()
            return activity.changes_since(last_seen)

        def discard_activity(request: gr.Request):
            with self._lock:
                self._sessions.pop(request.session_hash, None)

        timer.tick(
            refresh_activity, versions, [console, progress, versions],
            queue=False, show_progress="hidden", api_visibility="private",
        )
        app.unload(discard_activity)
