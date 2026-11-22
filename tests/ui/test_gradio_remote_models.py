import asyncio
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio", minversion="6.17.3")
from gradio.state_holder import SessionState

from tts_audiobook_tool import web_remote_panel
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_model_setup import WebModelSetupError
from tts_audiobook_tool.web_remote_models import WebRemoteTtsError


def test_remote_panel_requires_refresh_before_a_changed_url_can_render(monkeypatch):
    model = TtsModelType.require_by_id("chatterbox_audiocpp")
    requested = []

    def probe(endpoint):
        requested.append(endpoint)
        return RemoteTtsSnapshot(TtsBackendKind.AUDIO_CPP, candidates=((model, "server-alias"),))

    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", probe)
    monkeypatch.setattr(web_remote_panel, "remote_client_python", lambda: Path("venv-remote/bin/python"))

    with gr.Blocks(analytics_enabled=False) as app:
        panel = web_remote_panel.RemoteModelPanel()
        panel.bind(event_options={"api_visibility": "private"})
    state = SessionState(app)

    async def run(component, event, *inputs):
        fn = next(fn for fn in app.fns.values() if (component._id, event) in fn.targets)
        result = await app.process_api(fn, list(inputs), state)
        return dict(zip(fn.outputs, result["data"], strict=True))

    async def exercise():
        first_url = "http://first.example"
        second_url = "http://second.example"
        discovered = await run(panel.refresh, "click", first_url, None)
        old_choice = discovered[panel.model]["value"]
        first = panel.selection_for(first_url, state[panel.state._id])
        assert first.server_model_id == "server-alias"
        assert discovered[panel.unload]["interactive"] is True
        assert discovered[panel.client_status]["visible"] is True
        with pytest.raises(WebRemoteTtsError, match="current server URL"):
            panel.selection_for(second_url, state[panel.state._id])

        reset = await run(panel.endpoint, "input", second_url, None)
        assert reset[panel.model]["choices"] == []
        assert reset[panel.unload]["interactive"] is False
        with pytest.raises(WebRemoteTtsError, match="current server URL"):
            panel.selection_for(first_url, state[panel.state._id])

        discovered_again = await run(panel.refresh, "click", second_url, None)
        assert discovered_again[panel.model]["value"] != old_choice
        second = panel.selection_for(second_url, state[panel.state._id])
        assert second.server_model_id == first.server_model_id
        assert second.endpoint != first.endpoint

    asyncio.run(exercise())
    assert requested == ["http://first.example", "http://second.example"]


def test_remote_client_install_action_reports_progress_and_recovers_after_cancel(monkeypatch):
    calls = []

    def install(*, on_progress, cancel_event):
        calls.append(cancel_event.is_set())
        on_progress("Prepare the launcher marker")
        if cancel_event.is_set():
            raise WebModelSetupError("Model setup was stopped.")
        return Path("venv-remote-tts")

    monkeypatch.setattr(web_remote_panel, "install_remote_client", install)
    monkeypatch.setattr(web_remote_panel, "remote_client_install_supported", lambda: True)
    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", lambda *_args: pytest.fail("Unexpected server probe"))
    with gr.Blocks(analytics_enabled=False) as app:
        panel = web_remote_panel.RemoteModelPanel()
        panel.bind(event_options={"api_visibility": "private"})

    state = SessionState(app)
    request = gr.Request(session_hash="test-remote-install")
    start = next(fn for fn in app.fns.values() if (panel.install._id, "click") in fn.targets)
    stop = next(fn for fn in app.fns.values() if (panel.cancel_install._id, "click") in fn.targets)

    async def run_install(cancel=False):
        begin = await app.process_api(start, [], state, request=request)
        assert begin["data"][1]["interactive"] is False
        assert begin["data"][2]["interactive"] is True
        if cancel:
            await app.process_api(stop, [], state, request=request)
        result = await app.process_api(start, [], state, request=request, iterator=begin["iterator"])
        await app.process_api(start, [], state, request=request, iterator=result["iterator"])
        return result["data"]

    async def exercise():
        cancelled = await run_install(cancel=True)
        assert "stopped" in cancelled[0]["value"]
        assert cancelled[1]["interactive"] is True
        assert cancelled[2]["interactive"] is False
        completed = await run_install()
        assert "ready" in completed[0]["value"]
        assert completed[1]["interactive"] is False
        assert completed[2]["interactive"] is False

    asyncio.run(exercise())
    assert calls == [True, False]
