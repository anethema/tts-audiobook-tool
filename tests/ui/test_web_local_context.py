import json

import pytest
import requests

from tts_audiobook_tool import web_local_tagger as tagger


_INPUT_OVERFLOW = {"error": "request (18000 tokens) exceeds the available context size (16384 tokens)"}


def _done(content, reason="stop"):
    return {"message": {"content": content}, "done": True, "done_reason": reason}


def _serve(monkeypatch, chat_replies):
    replies = iter(chat_replies)
    chats = []

    def post(_http, url, *, json: dict, **_options):
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response._content_consumed = True
        if url.endswith("/api/chat"):
            chats.append(json)
            response.status_code, response._content = next(replies)
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    return chats


def _response(status, *frames):
    return status, b"\n".join(json.dumps(frame).encode() for frame in frames)


@pytest.mark.parametrize("limited", [
    _response(400, _INPUT_OVERFLOW),
    _response(200, {**_INPUT_OVERFLOW, "status": 400}),
    _response(200, _done("partial tags", "length")),
])
def test_entire_chunk_retried_once_then_next_chunk_starts_small_again(monkeypatch, limited):
    chats = _serve(monkeypatch, [limited, _response(200, _done("complete")), _response(200, _done("next"))])
    log = []
    with tagger.local_qwen(log.append) as local:
        assert local.send("instructions", "entire quoted scene").reply == "complete"
        assert local.send("instructions", "next scene").reply == "next"

    assert [chat["options"]["num_ctx"] for chat in chats] == [16_384, 40_960, 16_384]
    assert chats[1]["messages"] == chats[0]["messages"]
    assert chats[0]["messages"][-1]["content"] == "entire quoted scene"
    assert all(chat["truncate"] is False and chat["shift"] is False for chat in chats)
    assert all(chat["format"] == "json" and chat["think"] == "medium" for chat in chats)
    assert sum("Retrying with 40,960" in line for line in log) == 1


def test_exhausted_larger_context_reports_an_error_and_discards_partial_output(monkeypatch):
    limited = _response(200, _done("partial", "length"))
    chats = _serve(monkeypatch, [limited, limited])
    with tagger.local_qwen(lambda _line: None) as local:
        with pytest.raises(tagger.LocalTaggerError, match="40,960-token context"):
            local.send("instructions", "scene")
    assert len(chats) == 2


@pytest.mark.parametrize("unrelated", [
    _response(500, {"error": "failed to allocate memory for context"}),
    _response(200, {"error": "runner stopped unexpectedly"}),
    (503, b"unavailable"),
])
def test_ordinary_failures_do_not_trigger_an_expensive_retry(monkeypatch, unrelated):
    chats = _serve(monkeypatch, [unrelated])
    with tagger.local_qwen(lambda _line: None) as local:
        with pytest.raises(tagger.LocalTaggerError):
            local.send("instructions", "scene")
    assert len(chats) == 1
