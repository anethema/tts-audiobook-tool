import pytest

from tts_audiobook_tool import app_support
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.dots_base_model import DotsBaseModel
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.web_model_options import model_options_for, validate_model_options, voice_presets_for


def test_breeze_uses_catalog_defaults_and_writes_native_types():
    model_id = "breeze_tts_2_local"
    options = {item.key: item for item in model_options_for(model_id)}
    assert options["temperature"].default == 0.9
    assert options["top_k"].default == 50
    changed = validate_model_options(model_id, {"top_k": 25.0, "temperature": 0.9})
    assert changed == {"top_k": 25}
    project = Project(tts_model_type=model_id)
    for name, value in changed.items():
        project.set_model_setting(model_id, name, value)
    assert project.get_model_setting(model_id, "top_k") == 25


def test_fish_choice_is_typed_and_out_of_range_values_are_rejected():
    model_id = "fish_s2_local"
    assert validate_model_options(model_id, {"compile_enabled": False}) == {"compile_enabled": False}
    for bad in ({"compile_enabled": "False"}, {"top_k": 2.5}, {"temperature": float("nan")},
                {"temperature": 10.0}, {"unrecognized": "value"}):
        with pytest.raises(ValueError):
            validate_model_options(model_id, bad)


def test_qwen_stock_choices_resolve_model_type_and_reject_paths():
    model_id = "qwen3tts_local"
    target = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert validate_model_options(model_id, {"target": target, "temperature": None}) == {
        "target": target, "model_type": "voice_design"
    }
    with pytest.raises(ValueError):
        validate_model_options(model_id, {"target": "/tmp/untrusted"})


def test_qwen_custom_voice_choices_set_native_type_and_speaker_inventory():
    model_id = "qwen3tts_local"
    option = next(item for item in model_options_for(model_id) if item.key == "target")
    custom_targets = [target for target in Qwen3BaseModel.PRESET_REPO_IDS if target.endswith("-CustomVoice")]
    assert custom_targets and set(custom_targets) <= {value for _, value in option.choices}
    overrides = validate_model_options(model_id, {"target": custom_targets[0]})
    assert overrides == {"target": custom_targets[0], "model_type": "custom_voice"}
    assert "vivian" in voice_presets_for(model_id)
    with pytest.raises(ValueError, match="Rolling continuation requires"):
        validate_model_options(model_id, {"target": custom_targets[0], "rolling_cont": 1})


def test_pocket_chatterbox_and_dots_variants_use_native_settings():
    pocket = "pocket_local"
    chatterbox = "chatterbox_local"
    dots = "dots_local"
    assert voice_presets_for(pocket) == tuple(PocketBaseModel.PREDEFINED_VOICES)
    assert validate_model_options(pocket, {"model_code": PocketBaseModel.LANGUAGES[0]}) == {
        "model_code": PocketBaseModel.LANGUAGES[0],
    }
    overrides = validate_model_options(chatterbox, {"type": ChatterboxType.TURBO.id})
    project = Project(tts_model_type=chatterbox)
    project.set_model_setting(chatterbox, "type", overrides["type"])
    assert project.get_model_setting(chatterbox, "type") is ChatterboxType.TURBO
    assert validate_model_options(dots, {"target": DotsBaseModel.MF_1STEP_REPO_ID, "compile": False}) == {
        "target": DotsBaseModel.MF_1STEP_REPO_ID, "compile": False,
    }
