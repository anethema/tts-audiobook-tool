import asyncio
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

gr = pytest.importorskip("gradio", minversion="6.17.3")
from gradio.state_holder import SessionState

from tts_audiobook_tool import web_chat, web_chat_panel
from tts_audiobook_tool.web_render import RenderVoice


def _write_clip(path):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 800)


def test_microphone_draft_echo_audio_and_chat_lifecycle(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(uploads))
    microphone_clip = uploads / "message.wav"
    _write_clip(microphone_clip)
    available = [("Example speech model", "venv-example")]
    render_calls = []
    microphone_calls = []

    def transcribe(path, *, language_code):
        microphone_calls.append((Path(path), language_code))
        return "Recognized draft."

    def render(model, reply, voice, **settings):
        render_calls.append((model, reply, voice, settings))
        clip = tmp_path / "reply.wav"
        _write_clip(clip)
        return str(clip)

    monkeypatch.setattr(web_chat, "transcribe_model_reference", transcribe)
    monkeypatch.setattr(web_chat_panel, "render_model_sample", render)
    monkeypatch.setattr(web_chat.requests.Session, "post", lambda *_args, **_kw: pytest.fail("Unexpected provider call"))

    with gr.Blocks(analytics_enabled=False) as app:
        key = gr.Textbox(type="password")
        model = gr.Textbox()
        effort = gr.Textbox()
        endpoint = gr.Textbox()
        voice_name = gr.Textbox()
        voice_panel = SimpleNamespace(inputs=[voice_name], snapshot=lambda name: RenderVoice(name=name, pan=0.2))
        model_settings = gr.State({"venv-example": {"seed": 42}})
        panel = web_chat_panel.VoiceChatPanel(available, available[0][1])
        panel.bind(
            api_key=key, llm_model=model, effort=effort, endpoint=endpoint, voice_panel=voice_panel,
            model_settings=model_settings, event_options={"api_visibility": "private", "concurrency_id": "local-work"},
        )

    app.has_launched = True
    state = SessionState(app)
    request = gr.Request(session_hash="voice-chat-test")

    def event(name):
        return next(fn for fn in app.fns.values() if fn.name == name)

    async def run(name, *inputs):
        fn = event(name)
        result = await app.process_api(fn, list(inputs), state, request=request)
        frames = []
        while True:
            if result["is_generating"] or not frames:
                frames.append(dict(zip(fn.outputs, result["data"], strict=True)))
            if not result["is_generating"]:
                return frames
            result = await app.process_api(fn, list(inputs), state, request=request, iterator=result["iterator"])

    async def exercise():
        recorded = {"path": str(microphone_clip), "meta": {"_type": "gradio.FileData"}}
        assert set(event("transcribe_recording").targets) == {
            (panel.microphone._id, "stop_recording"), (panel.microphone._id, "upload"),
        }
        start, finish = await run("transcribe_recording", recorded, "fr", None)
        assert start[panel.message] == gr.update(value="", interactive=False)
        assert finish[panel.message] == gr.update(value="Recognized draft.", interactive=True)

        send_values = (
            None, "Edited draft.", "sk-never-sent", "", "", "", "", "", True, True,
            "venv-example", "fr", None, "Narrator voice",
        )
        frames = await run("send_message", *send_values)
        assert state[panel.state._id].turns == (("Edited draft.", "Edited draft."),)
        final = frames[-1]
        assert final[panel.audio]["visible"] is True
        assert Path(final[panel.audio]["value"]["path"]).is_file()
        assert "download" in panel.audio.buttons
        assert final[panel.message] == gr.update(value="", interactive=True)
        assert not (tmp_path / "reply.wav").exists()

        (ended,) = await run("end_chat", None)
        assert state[panel.state._id].ended
        assert ended[panel.send]["interactive"] is False
        (rejected,) = await run("send_message", *send_values)
        assert rejected[panel.status]["visible"] is True
        await run("reset_chat")
        assert state[panel.state._id] == web_chat.ChatState()

    asyncio.run(exercise())
    assert microphone_calls == [(microphone_clip, "fr")]
    model, reply, voice, settings = render_calls[0]
    assert (model, reply, voice.name, voice.pan) == ("venv-example", "Edited draft.", "Narrator voice", 0.2)
    assert settings["language_code"] == "fr"
    assert settings["model_options"] == {"seed": 42}
    assert not settings["cancel_event"].is_set()
