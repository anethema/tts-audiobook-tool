from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

import launch
from tts_audiobook_tool import web_models, web_remote_models, web_render
from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.tts import Tts, WEB_REMOTE_ENTRY_ID_ENV
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_backend import BookSession


AUDIO_HEALTH = {"status": "ok", "backend": "cpu", "models": 2, "ui": False, "ui_management": False}
AUDIO_MODEL = {"family": "chatterbox", "task": "clon", "mode": "offline",
               "session_options": {"chatterbox.multilingual_t3": "v3"}}
SGL_ID = "org/higgs-v3"


def _mock_server(monkeypatch, format_response):
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(format_response), **kwargs))


def test_discovery_keeps_exact_model_pairs_scoped_to_the_verified_endpoint(monkeypatch):
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "http://console.example")
    calls = []
    ids = ["operator's first model", "same family / second"]

    def respond(request):
        calls.append((request.url.host, request.url.path))
        if request.url.path == "/tts/health":
            return httpx.Response(200, json=AUDIO_HEALTH)
        assert request.url.path == "/tts/v1/models"
        assert dict(request.url.params) == {"include_session_options": "true"}
        return httpx.Response(200, json={"data": [{"id": name, **AUDIO_MODEL} for name in ids]})

    _mock_server(monkeypatch, respond)
    first_endpoint = "http://first.example/tts"
    other_endpoint = "http://other.example/tts"
    first = web_remote_models.discover_remote_models(first_endpoint + "//")
    assert first.selected is None
    assert [(model.catalog_id, model.server_model_id) for model in first.candidates] == [
        ("chatterbox_audiocpp", name) for name in ids
    ]
    chosen = web_remote_models.select_remote_model(first_endpoint, first, first.candidates[1].key)
    assert web_remote_models.resolve_remote_selection(first_endpoint, chosen).server_model_id == ids[1]
    with pytest.raises(web_remote_models.WebRemoteTtsError, match="current server URL"):
        web_remote_models.resolve_remote_selection(other_endpoint, chosen)
    refreshed = web_remote_models.discover_remote_models(first_endpoint, chosen)
    assert refreshed.selected == chosen.selected
    other = web_remote_models.discover_remote_models(other_endpoint, chosen)
    assert other.selected is None
    with pytest.raises(web_remote_models.WebRemoteTtsError, match="current server’s discovered list"):
        web_remote_models.select_remote_model(other_endpoint, other, chosen.selected.key)
    ids[:] = ["replacement alias"]
    replaced = web_remote_models.discover_remote_models(first_endpoint, chosen)
    assert len(replaced.candidates) == 1 and replaced.selected is None
    assert [host for host, _ in calls] == ["first.example"] * 4 + ["other.example"] * 2 + ["first.example"] * 2
    assert RemoteTtsDiscovery.get_base_url() == "http://console.example"


def test_explicit_unload_rechecks_audio_cpp_and_honors_changed_endpoints(monkeypatch):
    state = {"backend": "audio.cpp"}
    unload_calls = []
    endpoint = "https://server.example"

    def respond(request):
        if state["backend"] == "audio.cpp":
            payload = AUDIO_HEALTH if request.url.path == "/health" else {"data": [{"id": "voice", **AUDIO_MODEL}]}
        else:
            payload = {"status": "healthy"} if request.url.path == "/health" else {"data": [{"id": SGL_ID}]}
        return httpx.Response(200, json=payload)

    _mock_server(monkeypatch, respond)
    monkeypatch.setattr(AudioCppUtil, "unload_all_models", lambda target: unload_calls.append(target))
    config = web_remote_models.discover_remote_models(endpoint)
    web_remote_models.unload_audio_cpp_models(endpoint + "/", config)
    with pytest.raises(web_remote_models.WebRemoteTtsError, match="current server URL"):
        web_remote_models.unload_audio_cpp_models("https://changed.example", config)
    state["backend"] = "SGL-Omni"
    with pytest.raises(web_remote_models.WebRemoteTtsError, match="verified as audio.cpp"):
        web_remote_models.unload_audio_cpp_models(endpoint, config)
    assert unload_calls == [endpoint]


def test_rejects_credentials_before_probing(monkeypatch):
    monkeypatch.setattr(RemoteTtsDiscovery, "_probe", lambda *_args: pytest.fail("Unexpected remote probe"))
    with pytest.raises(web_remote_models.WebRemoteTtsError, match="without credentials") as failure:
        web_remote_models.discover_remote_models("https://username:example-token@test.example")
    assert "example-token" not in str(failure.value)


def test_remote_client_uses_launcher_marker_and_native_venv_python(tmp_path, monkeypatch):
    candidate_names = ("venv-remote-tts", "venv-remote", "venv-other")
    platform_python = Path("Scripts/python.exe") if sys.platform == "win32" else Path("bin/python")
    for name in candidate_names:
        venv = tmp_path / name
        (venv / platform_python).parent.mkdir(parents=True)
        (venv / platform_python).touch()
        (venv / "pyvenv.cfg").touch()
    probes = []

    def marker(path):
        probes.append(Path(path).name)
        return Path(path).name == "venv-remote"

    monkeypatch.setattr(launch, "has_remote_client_marker", marker)
    assert web_remote_models.remote_client_python(tmp_path) == tmp_path / "venv-remote" / platform_python
    assert probes == ["venv-remote-tts", "venv-remote"]


@pytest.mark.parametrize("action", ["sample", "book"])
def test_remote_render_dispatches_through_the_marked_worker_with_an_exact_live_pair(monkeypatch, tmp_path, action):
    selection = web_remote_models.RemoteTtsSelection("https://tts.example", "higgs_v3_sglomni", SGL_ID)
    remote_python = tmp_path / "venv-remote" / "bin" / "python"
    sample = tmp_path / "mock-render.wav"
    sample.touch()
    probes = []
    launched = []
    received = []

    def snapshot(*, force):
        probes.append((RemoteTtsDiscovery.get_base_url(), force))
        model = TtsModelType.require_by_id(selection.catalog_id)
        return RemoteTtsSnapshot(TtsBackendKind.SGL_OMNI, candidates=((model, selection.server_model_id),))

    def render(*_args, remote_selection, **_kwargs):
        assert os.environ[WEB_REMOTE_ENTRY_ID_ENV] == selection.server_model_id
        received.append(remote_selection)
        return str(sample)

    def run(command, **_options):
        launched.append(command[0])
        directory = Path(command[-1])
        data = json.loads((directory / "input.json").read_text(encoding="utf-8"))
        assert data["remote_selection"] == asdict(selection)
        web_models._worker(directory)
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setenv(WEB_REMOTE_ENTRY_ID_ENV, "outer-context")
    monkeypatch.setattr(RemoteTtsDiscovery, "_base_url", "https://outer.example")
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", snapshot)
    monkeypatch.setattr(Tts, "init_local_model_type", lambda: probes.append("init"))
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: True)
    monkeypatch.setattr(web_models, "remote_client_python", lambda _root: remote_python)
    monkeypatch.setattr(web_models, "_python_for", lambda *_args: pytest.fail("Local model selected unexpectedly"))
    monkeypatch.setattr(web_models.subprocess, "Popen", run)
    monkeypatch.setattr(web_models.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", lambda: None)
    monkeypatch.setattr(web_render, f"render_{action}", render)
    if action == "sample":
        output = web_models.render_model_sample(
            web_remote_models.REMOTE_MODEL_ID, "Hello.", web_render.RenderVoice(), remote_selection=selection,
        )
    else:
        book = Book([BookSection([PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])])])
        output = web_models.render_model_book(
            web_remote_models.REMOTE_MODEL_ID, BookSession(book, str(tmp_path)), {}, remote_selection=selection,
        )
    assert output == str(sample)
    assert launched == [str(remote_python)]
    assert probes == ["init", (selection.endpoint, True)]
    assert received == [selection]
    assert RemoteTtsDiscovery.get_base_url() == "https://outer.example"
    assert os.environ[WEB_REMOTE_ENTRY_ID_ENV] == "outer-context"


def test_remote_worker_rejects_a_replaced_server_model_before_rendering(monkeypatch, tmp_path):
    current = TtsModelType.require_by_id("higgs_v3_sglomni")
    selected = web_remote_models.RemoteTtsSelection("https://tts.example", current.id, SGL_ID)
    monkeypatch.setattr(RemoteTtsDiscovery, "refresh", lambda **_: RemoteTtsSnapshot(
        TtsBackendKind.SGL_OMNI, candidates=((current, "org/replacement-higgs"),),
    ))
    monkeypatch.setattr(Tts, "init_local_model_type", lambda: None)
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: True)
    monkeypatch.setattr(web_models.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", lambda: None)
    monkeypatch.setattr(web_render, "render_sample", lambda *_args, **_kwargs: pytest.fail("Stale remote pair rendered"))
    payload = {"action": "sample", "remote_selection": asdict(selected)}
    (tmp_path / "input.json").write_text(json.dumps(payload), encoding="utf-8")
    web_models._worker(tmp_path)
    assert "unavailable" in json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["error"]


def test_remote_render_needs_a_verified_pair_and_installed_client(monkeypatch):
    monkeypatch.setattr(web_models, "remote_client_python", lambda _root: None)
    monkeypatch.setattr(web_models.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Unexpected worker launch"))
    model_id = web_remote_models.REMOTE_MODEL_ID
    voice = web_render.RenderVoice()
    with pytest.raises(web_models.ModelRunError, match="verified remote model"):
        web_models.render_model_sample(model_id, "Hello.", voice)
    selection = web_remote_models.RemoteTtsSelection("https://tts.example", "higgs_v3_sglomni", SGL_ID)
    with pytest.raises(web_models.ModelRunError, match="installed remote speech client"):
        web_models.render_model_sample(model_id, "Hello.", voice, remote_selection=selection)
