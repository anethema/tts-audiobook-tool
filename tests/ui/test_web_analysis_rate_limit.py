import requests

from tts_audiobook_tool import web_analysis_rate_limit as limits
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult
from tts_audiobook_tool.web_backend import _SanitizedHttpError


def test_retry_delay_keeps_only_valid_provider_timing():
    headers = requests.structures.CaseInsensitiveDict({"Retry-After": "12", "Retry-After-Ms": "750"})
    assert limits.retry_after_seconds(headers) == 0.75
    headers["Retry-After-Ms"] = "unknown"
    assert limits.retry_after_seconds(headers) == 12
    assert limits.retry_after_seconds({"retry-after": "inf"}) is None


def test_rate_limit_retries_are_bounded_and_billing_errors_return_immediately(monkeypatch):
    now = 100.0
    def advance(seconds):
        nonlocal now
        now += seconds

    monkeypatch.setattr(limits, "monotonic", lambda: now)
    monkeypatch.setattr(limits, "sleep", advance)
    limiter = limits.RequestRateLimiter()
    calls = []
    logs = []

    def response(code="", delay=2):
        calls.append(now)
        return LlmRequestResult("", "", 0, error=_SanitizedHttpError(429, code, delay))

    assert limiter.send(response, on_log=logs.append).error is not None
    assert calls == [100, 102, 104, 106]
    assert len(logs) == 3 and limiter.requests_sent == 4 and limiter.peak_in_flight == 1

    reply = limiter.send(lambda: response("insufficient_quota"), on_log=logs.append)
    assert reply.error.code == "insufficient_quota" and len(calls) == 5
    assert len(logs) == 3
