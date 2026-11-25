import json

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from tts_audiobook_tool import web_local_tagger as tagger


def _mock_ollama(monkeypatch, chat_lines, *, on_chat=None):
    calls = []

    def post(session, url, **kwargs):
        route = url.rsplit("/", 1)[-1]
        calls.append(route)
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        if route == "chat":
            assert kwargs["stream"] is True and kwargs["json"]["stream"] is True
            if on_chat is not None:
                on_chat(kwargs)
            response.iter_lines = chat_lines
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    return calls


def test_local_chat_allows_slow_prefill_and_a_long_active_stream(monkeypatch):
    now = 0
    monkeypatch.setattr(tagger, "monotonic", lambda: now)

    def prefill(kwargs):
        nonlocal now
        connect, idle = kwargs["timeout"]
        assert 0 < connect <= 60
        assert idle == 60 * 60
        now += 1801

    def chat_lines():
        nonlocal now
        yield b'{"message":{"content":"first"},"done":false}'
        now += 1801
        yield b'{"message":{"content":" last"},"done":true}'

    _mock_ollama(monkeypatch, chat_lines, on_chat=prefill)
    with tagger.local_qwen(lambda _line: None) as model:
        result = model.send("system", "prompt")
    assert result.reply == "first last"
    assert result.elapsed == 3602


@pytest.mark.parametrize("error, message", [
    (requests.ReadTimeout("private details"), "Ollama sent no data for 60 minutes"),
    (requests.ConnectTimeout("private details"), "Cannot reach Ollama"),
])
def test_timeout_before_chat_headers_explains_the_problem(monkeypatch, error, message):
    def fail(_kwargs):
        raise error

    calls = _mock_ollama(monkeypatch, None, on_chat=fail)
    with pytest.raises(tagger.LocalTaggerError) as caught:
        with tagger.local_qwen(lambda _line: None) as model:
            model.send("system", "prompt")

    assert message in str(caught.value)
    assert "retry" in str(caught.value).lower()
    assert "private details" not in str(caught.value)
    assert calls == ["show", "chat", "generate"]


@pytest.mark.parametrize("options, expected", [({}, "medium"), ({"thinking": "off"}, False)])
def test_local_qwen_sends_selected_thinking_mode(monkeypatch, options, expected):
    def chat_lines():
        yield b'{"message":{"content":"answer"},"done":true}'

    def check_thinking(kwargs):
        actual = kwargs["json"]["think"]
        assert actual == expected
        assert type(actual) is type(expected)

    _mock_ollama(monkeypatch, chat_lines, on_chat=check_thinking)
    with tagger.local_qwen(lambda _line: None, **options) as model:
        assert model.send("system", "prompt").reply == "answer"


def test_local_progress_arrives_before_completion_and_keeps_exact_reply(monkeypatch):
    now = 100.0
    monkeypatch.setattr(tagger, "monotonic", lambda: now)
    activity = []

    def chat_lines():
        nonlocal now
        yield b""
        yield b'{"message":{"content":"","thinking":"private reasoning"},"done":false}'
        assert "reasoning" in activity[-1].lower()
        assert "private" not in activity[-1]
        yield b'{"message":{"content":"first "},"done":false}'
        first_update = activity[-1]
        assert first_update == "Qwen is writing tags · 6 reply characters · 0s"
        now += 0.25
        yield b'{"message":{"content":"private "},"done":false}'
        assert activity[-1] == first_update
        now += 2
        yield json.dumps({"message": {"content": "café"}, "done": False}).encode()
        assert activity[-1] == "Qwen is writing tags · 18 reply characters · 2s"
        yield b'{"message":{"content":"!"},"done":true,"done_reason":"stop","prompt_eval_count":42,"eval_count":6}'

    calls = _mock_ollama(monkeypatch, chat_lines)
    with tagger.local_qwen(lambda _line: None, on_activity=activity.append) as model:
        result = model.send("system", "prompt")

    assert calls == ["show", "chat", "generate"]
    assert result.reply == "first private café!" and result.reasoning == ""
    assert result.input_tokens == 42 and result.output_tokens == 6
    assert result.finish_reason == "stop" and result.elapsed == 2.25
    assert all("private" not in update for update in activity)


@pytest.mark.parametrize("end, expected", [
    (None, "before reporting completion"),
    (b"not JSON", "incomplete chat response"),
    (b'{"done":true,"message":{"content":17}}', "incomplete chat response"),
    (b'{"error":"private failure details"}', "failed during streaming"),
    (requests.ConnectionError("connection closed"), "response was interrupted"),
    (requests.ReadTimeout("private timeout"), "Ollama sent no data for 60 minutes"),
    # Requests wraps urllib3 read timeouts as ConnectionError during body iteration.
    (requests.ConnectionError(ReadTimeoutError(None, None, "private timeout")), "Ollama sent no data for 60 minutes"),
])
def test_local_stream_failure_discards_partial_reply_and_unloads(monkeypatch, end, expected):
    def chat_lines():
        yield b'{"message":{"content":"partial"},"done":false}'
        if isinstance(end, Exception):
            raise end
        if end is not None:
            yield end

    calls = _mock_ollama(monkeypatch, chat_lines)
    with pytest.raises(tagger.LocalTaggerError, match=expected) as caught:
        with tagger.local_qwen(lambda _line: None) as model:
            model.send("system", "prompt")

    if isinstance(end, requests.RequestException):
        assert "retry" in str(caught.value).lower()
    assert "partial" not in str(caught.value) and "private" not in str(caught.value)
    assert calls == ["show", "chat", "generate"]


def test_stream_progress_counts_completed_fields_without_revealing_labels(monkeypatch):
    now = 100.0
    monkeypatch.setattr(tagger, "monotonic", lambda: now)
    activity = []
    fragments = [
        '{"speakers":{"1":"AL',
        'ICE","2":"BO',
        'B","foo":"ignored","3":" ","4":false},"delivery":{"2":{"direction":"qui',
        'etly"},"5":{}}}',
    ]

    def chat_lines():
        nonlocal now
        for fragment, labels in zip(fragments, (0, 1, 2, 2), strict=True):
            now += 2
            yield json.dumps({"message": {"content": fragment}, "done": False}).encode()
            assert f"{labels} speaker label" in activity[-1]
        assert "1 delivery cue drafted" in activity[-1]
        yield b'{"message":{"content":""},"done":true}'

    _mock_ollama(monkeypatch, chat_lines)
    with tagger.local_qwen(lambda _line: None, on_activity=activity.append) as model:
        result = model.send("system", "prompt")

    assert result.reply == "".join(fragments)
    assert all("ALICE" not in update and "quietly" not in update for update in activity)
    assert "0 delivery cues drafted" in activity[-2]
