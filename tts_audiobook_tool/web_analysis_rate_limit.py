"""Coordinate backoff across a book's concurrent remote analysis requests."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite
from threading import Lock
from time import monotonic, sleep

import requests

from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult


def retry_after_seconds(headers) -> float | None:
    """Keep only a numeric delay when discarding the provider's raw error response."""
    for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(key)
        if not isinstance(raw, str):
            continue
        try:
            seconds = float(raw) * scale
        except ValueError:
            if key != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(raw)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = (date - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                continue
        if isfinite(seconds) and seconds >= 0:
            return seconds
    return None


class RequestRateLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._next_allowed = 0.0
        self._spread_until = 0.0
        self._active = 0
        self.peak_in_flight = 0
        self.requests_sent = 0

    def _begin(self) -> None:
        while True:
            with self._lock:
                now = monotonic()
                wait = self._next_allowed - now
                if wait <= 0:
                    # Stagger workers after a shared cooldown instead of sending another burst.
                    if now < self._spread_until:
                        self._next_allowed = now + 0.15
                    self._active += 1
                    self.peak_in_flight = max(self.peak_in_flight, self._active)
                    self.requests_sent += 1
                    return
            sleep(min(wait, 60))

    def send(self, request: Callable[[], LlmRequestResult], *, on_log: Callable[[str], None]) -> LlmRequestResult:
        max_retries = 3
        retries = 0
        while True:
            self._begin()
            try:
                reply = request()
            finally:
                with self._lock:
                    self._active -= 1
            error = reply.error
            if (not isinstance(error, requests.HTTPError) or error.response is None
                    or error.response.status_code != 429 or getattr(error, "code", "") == "insufficient_quota"
                    or retries == max_retries):
                return reply
            delay = getattr(error, "retry_after", None)
            if delay is None:
                delay = 5 * 2 ** retries
            retries += 1
            with self._lock:
                self._next_allowed = max(self._next_allowed, monotonic() + delay)
                self._spread_until = self._next_allowed + 3
            on_log(f"Provider rate limit. Retrying in {delay:g}s ({retries}/{max_retries}).\n")
