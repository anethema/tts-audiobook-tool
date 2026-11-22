import json
import os
from pathlib import Path
import stat

import pytest

from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, AAC_BITRATES
from tts_audiobook_tool.web_runtime_settings import (
    WebRuntimeSettings, WebRuntimeSettingsError, load_runtime_settings, save_runtime_settings,
)


def test_explicit_whisper_cache_is_required_only_when_validating(monkeypatch, tmp_path):
    from faster_whisper import utils
    from tts_audiobook_tool.stt import Stt

    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: False)
    lookups = []

    def lookup(variant, *, local_files_only):
        lookups.append((variant, local_files_only))
        return str(tmp_path)

    monkeypatch.setattr(utils, "download_model", lookup)
    settings = WebRuntimeSettings(whisper_variant=SttVariant.LARGE_V3.id, whisper_device=SttConfig.CPU_INT8FLOAT32.id)
    settings.check_validation_availability(settings.generation_prefs(str(tmp_path), validate_audio=False))
    assert not lookups
    prefs = settings.generation_prefs(str(tmp_path), validate_audio=True)
    with pytest.raises(WebRuntimeSettingsError, match="incomplete"):
        settings.check_validation_availability(prefs)
    assert lookups == [(SttVariant.LARGE_V3.id, True)]


def test_explicit_cuda_whisper_selection_reports_unavailable_device(monkeypatch, tmp_path):
    from tts_audiobook_tool.stt import Stt
    from tts_audiobook_tool.system_support.gpu_caps_util import GpuCapsUtil

    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: False)
    monkeypatch.setattr(GpuCapsUtil, "has_ctranslate2_float16_gpu", lambda: False)
    settings = WebRuntimeSettings(whisper_device=SttConfig.CUDA_FLOAT16.id)
    prefs = settings.generation_prefs(str(tmp_path), validate_audio=True)
    with pytest.raises(WebRuntimeSettingsError, match="CUDA float16 is unavailable"):
        settings.check_validation_availability(prefs)


def test_preferences_round_trip_without_console_writes_and_save_failure_keeps_old_values(monkeypatch, tmp_path):
    browser = tmp_path / "web-runtime-preferences.json"
    console = browser.with_name("prefs.json")
    console.write_text('{"console-only": "exact bytes"}', encoding="utf-8")
    assert load_runtime_settings(path=browser) == WebRuntimeSettings()
    selected_bitrate = next(item for item in AAC_BITRATES if item != AAC_BITRATE_DEFAULT)
    options = WebRuntimeSettings(
        force_cpu=True, whisper_variant=SttVariant.LARGE_V3.id, whisper_device=SttConfig.CPU_INT8FLOAT32.id,
        save_generation_logs=True, aac_bitrate=selected_bitrate,
    )
    save_runtime_settings(options, path=browser)
    assert load_runtime_settings(path=browser) == options
    assert console.read_text(encoding="utf-8") == '{"console-only": "exact bytes"}'
    if os.name != "nt":
        assert not stat.S_IMODE(browser.stat().st_mode) & 0o077
    assert set(json.loads(browser.read_text(encoding="utf-8"))["runtime"]) == {
        "force_cpu", "whisper_variant", "whisper_device", "save_generation_logs", "aac_bitrate",
    }

    def fail_replace(self, destination):
        raise OSError("test save interrupted")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(WebRuntimeSettingsError, match="Could not save"):
        save_runtime_settings(WebRuntimeSettings(), path=browser)
    assert load_runtime_settings(path=browser) == options
    assert sorted(tmp_path.iterdir()) == [console, browser]


def test_runtime_panel_restores_saved_fields_for_a_fresh_page(monkeypatch, tmp_path):
    import gradio as gr

    from tts_audiobook_tool import web_runtime_panel

    path = tmp_path / "prefs.json"
    monkeypatch.setattr(web_runtime_panel, "load_runtime_settings", lambda: load_runtime_settings(path=path))
    monkeypatch.setattr(web_runtime_panel, "save_runtime_settings", lambda settings: save_runtime_settings(settings, path=path))

    with gr.Blocks() as app:
        panel = web_runtime_panel.create_runtime_panel(gr.State(None), app=app, private_event={"api_visibility": "private"})
    assert len(panel.inputs) == 5
    assert panel.snapshot(False, "", "", False) == WebRuntimeSettings()
    handlers = {item.name: item.fn for item in app.fns.values()}
    selected_bitrate = next(item for item in AAC_BITRATES if item != AAC_BITRATE_DEFAULT)
    values = (True, SttVariant.LARGE_V3.id, SttConfig.CPU_INT8FLOAT32.id, True, selected_bitrate)
    assert handlers["save_preferences"](*values)["value"] == "Saved browser preferences."
    assert handlers["restore_preferences"]()[:5] == values
    assert "show_log" in handlers
