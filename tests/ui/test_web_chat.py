import json
import logging
from threading import Event

import pytest
import requests

from tts_audiobook_tool import web_chat
from tts_audiobook_tool.l import L


KEY = "sk-chat-test-secret"


@pytest.fixture(autouse=True)
def chat_logging(monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger("test_web_chat"), raising=False)


class Reply(requests.Response):
    def __init__(self, *, chunks=(), body=None, status=200):
        super().__init__()
        self.status_code = status
        self.chunks = chunks
        self._content = json.dumps(body).encode()
        self._content_consumed = True

    def iter_lines(self, **_kwargs):
        yield from self.chunks


def delta(text):
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}".encode()


def prepare(state=None, message="Hello.", **settings):
    defaults = dict(
        api_key=KEY, model="gpt-example", endpoint="https://api.openai.com/v1/chat/completions?version=1",
        effort="low", extra_json='{"max_completion_tokens": 200}', system_prompt="Speak briefly.",
        echo=False, stream=True,
    )
    return web_chat.prepare_chat_turn(state or web_chat.ChatState(), message, **{**defaults, **settings})


def test_streams_provider_reply_and_reuses_history_without_sending_credentials_in_messages(monkeypatch, caplog):
    responses = iter([
        Reply(chunks=(b": keep-alive", delta("Hello"), delta(" again."), b"data: [DONE]")),
        Reply(body={"choices": [{"message": {"content": "Of course."}, "finish_reason": "stop"}]}),
    ])
    requests_made = []

    def post(http, url, **kwargs):
        assert not http.trust_env
        assert kwargs["allow_redirects"] is False
        assert kwargs["headers"] == {"Authorization": f"Bearer {KEY}"}
        assert url.endswith("/v1/chat/completions?version=1")
        assert KEY not in json.dumps(kwargs["json"])
        requests_made.append(kwargs["json"])
        return next(responses)

    monkeypatch.setattr(requests.Session, "post", post)
    with caplog.at_level(logging.INFO):
        first = prepare()
        assert list(first.reply()) == ["Hello", "Hello again."]
        committed = first.commit("Hello again.")
        second = prepare(committed, "One more?", stream=False)
        assert list(second.reply()) == ["Of course."]

    assert requests_made[0]["store"] is False
    assert requests_made[0]["stream"] is True
    assert requests_made[0]["reasoning_effort"] == "low"
    assert requests_made[0]["max_completion_tokens"] == 200
    assert "temperature" not in requests_made[0]
    assert "stream" not in requests_made[1]
    assert requests_made[1]["messages"] == [
        {"role": "system", "content": "Speak briefly."},
        *committed.messages(), {"role": "user", "content": "One more?"},
    ]
    assert KEY not in repr(committed) + caplog.text
    assert "version=1" not in caplog.text
    with pytest.raises(web_chat.WebChatError, match="Start a new chat"):
        prepare(committed, system_prompt="A new instruction.")


def test_stream_does_not_publish_a_key_split_across_provider_events(monkeypatch, caplog):
    response = Reply(chunks=(delta("Safe text sk-chat-"), delta("test-secret"), b"data: [DONE]"))
    monkeypatch.setattr(requests.Session, "post", lambda *_args, **_kwargs: response)
    reply = prepare().reply()
    assert next(reply) == "Safe text "
    with pytest.raises(web_chat.WebChatError, match="invalid reply") as failure:
        next(reply)
    assert KEY not in str(failure.value) + caplog.text


def test_provider_error_body_does_not_reach_chat_or_logs(monkeypatch, caplog):
    response = Reply(status=401, body={"error": {"message": f"Rejected token {KEY}"}})
    monkeypatch.setattr(requests.Session, "post", lambda *_args, **_kwargs: response)
    with pytest.raises(web_chat.WebChatError, match="denied the request") as failure:
        list(prepare(stream=False).reply())
    assert KEY not in str(failure.value) + caplog.text


def test_stopping_stream_rolls_back_partial_turn(monkeypatch):
    response_closed = Event()

    class WaitingReply(Reply):
        def iter_lines(self, **_kwargs):
            yield delta("Partial reply")
            assert response_closed.wait(timeout=2)

        def close(self):
            response_closed.set()

    monkeypatch.setattr(requests.Session, "post", lambda *_args, **_kwargs: WaitingReply())
    active = prepare()
    reply = active.reply()
    assert next(reply) == "Partial reply"
    active.stop()
    with pytest.raises(web_chat.ChatStopped):
        next(reply)
    assert response_closed.wait(timeout=2)
    assert active.state.messages() == []


def test_echo_never_contacts_provider_and_key_placement_is_validated(monkeypatch):
    monkeypatch.setattr(requests.Session, "post", lambda *_args, **_kwargs: pytest.fail("Unexpected provider call"))
    repeated = prepare(message="  Repeat me. ", echo=True, model="", endpoint="", api_key="")
    assert list(repeated.reply()) == ["Repeat me."]
    assert repeated.commit("Repeat me.").turns == (("Repeat me.", "Repeat me."),)
    with pytest.raises(web_chat.WebChatError, match="API key field"):
        prepare(message=f"The token is {KEY}.")
    with pytest.raises(web_chat.WebChatError, match="HTTPS"):
        prepare(endpoint="http://example.com/chat/completions")
