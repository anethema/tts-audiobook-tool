"""Expose local model settings through the browser using the console catalog."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from tts_audiobook_tool import target_util
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.dots_base_model import DotsBaseModel, DotsCompileMode
from tts_audiobook_tool.tts_models.local_definition import LocalChoiceParameter, get_local_definition
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel, Qwen3ModelType
from tts_audiobook_tool.tts_models.vibevoice_base_model import VibeVoiceBaseModel


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str
    default: object
    minimum: int | float | None = None
    maximum: int | float | None = None
    integer: bool = False
    choices: tuple[tuple[str, object], ...] = ()
    info: str = ""


_LABELS = {
    "cfg": "CFG",
    "sr": "Sample rate",
    "top_k": "Top-K",
    "top_p": "Top-P",
    "use_fp16": "FP16",
    "num_step": "Diffusion steps",
    "rolling_cont": "Rolling continuation",
    "lora_strength": "LoRA strength",
}
_QWEN = "qwen3tts_local"
_LORA_MODELS = frozenset({"breeze_tts_2_local", "omnivoice_local", "vibevoice_local"})
_QWEN_TYPE_BY_SUFFIX = {
    "Base": Qwen3ModelType.BASE.value,
    "CustomVoice": Qwen3ModelType.CUSTOM_VOICE.value,
    "VoiceDesign": Qwen3ModelType.VOICE_DESIGN.value,
}
_QWEN_CHECKPOINTS = tuple(
    target for target in Qwen3BaseModel.PRESET_REPO_IDS if target.rsplit("-", 1)[-1] in _QWEN_TYPE_BY_SUFFIX
)
_QWEN_CUSTOM_VOICE_SPEAKERS = (
    "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian",
)


def voice_presets_for(catalog_id: str) -> tuple[str, ...]:
    """Built-in speakers available without loading a checkpoint."""
    if catalog_id == _QWEN:
        return _QWEN_CUSTOM_VOICE_SPEAKERS
    if catalog_id == "pocket_local":
        return tuple(PocketBaseModel.PREDEFINED_VOICES)
    return ()


def _checkpoint_option(default: str, choices: tuple[str, ...] | list[str]) -> ModelOption:
    return ModelOption("target", "Checkpoint", "", choices=(
        (f"Default ({default.split('/', 1)[-1]})", ""),
        *((target.split("/", 1)[-1], target) for target in choices if target != default),
    ))


def model_options_for(catalog_id: str) -> tuple[ModelOption, ...]:
    """Describe the supported editable options without loading a speech model."""
    definition = get_local_definition(catalog_id)
    labels = {control.parameter: control.label for control in definition.menu if control.label} if definition else {}
    result = []
    for key, parameter in (definition.parameters.items() if definition else ()):
        label = labels.get(key) or _LABELS.get(key) or key.replace("_", " ").capitalize()
        if isinstance(parameter, LocalChoiceParameter):
            choices = tuple((str(choice.label), value) for choice, value in zip(parameter.choices, parameter.values))
            result.append(ModelOption(key, label, parameter.typed_default, choices=choices))
        else:
            result.append(ModelOption(key, label, parameter.default, parameter.min, parameter.max,
                                      integer=parameter.type == "int"))
    if catalog_id in _LORA_MODELS:
        result.append(ModelOption(
            "lora_target", "LoRA adapter", "",
            info="Enter a local adapter directory or a Hugging Face repository. Leave blank to clear it.",
        ))
    if catalog_id == _QWEN:
        # The console derives the defaults from the checkpoint at runtime.
        result.extend((
            _checkpoint_option(Qwen3BaseModel.DEFAULT_REPO_ID, _QWEN_CHECKPOINTS),
            ModelOption("temperature", "Temperature", None, Qwen3BaseModel.TEMPERATURE_MIN,
                        Qwen3BaseModel.TEMPERATURE_MAX, info="Leave blank to use the checkpoint default."),
            ModelOption("top_p", "Top-P", None, 0.0, 1.0, info="Leave blank to use the checkpoint default."),
            ModelOption("top_k", "Top-K", None, 0, 1000, integer=True,
                        info="Leave blank to use the checkpoint default."),
            ModelOption("repetition_penalty", "Repetition penalty", None, 0.1, 3.0,
                        info="Leave blank to use the checkpoint default."),
            ModelOption("rolling_cont", "Rolling continuation", 0, 0,
                        Qwen3BaseModel.ROLLING_CONTINUATION_MAX_LENGTH, integer=True,
                        info="Requires a Base checkpoint and a batch size of one."),
        ))
    elif catalog_id == "pocket_local":
        languages = [("Select a Pocket model", "")]
        for code in PocketBaseModel.LANGUAGES:
            note = PocketBaseModel.LANGUAGE_NOTES.get(code)
            languages.append((f"{code} ({note})" if note else code, code))
        result.insert(0, ModelOption("model_code", "Pocket model", "", choices=tuple(languages)))
    elif catalog_id == "chatterbox_local":
        binding = REGISTRY.get(catalog_id, "type")
        result.insert(0, ModelOption(
            "type", "Chatterbox variant", binding.default, choices=tuple((item.label, item.id) for item in ChatterboxType),
        ))
    elif catalog_id == "dots_local":
        result[0:0] = [
            _checkpoint_option(DotsBaseModel.DEFAULT_REPO_ID, DotsBaseModel.PRESET_REPO_IDS),
            ModelOption(
                "compile", "Inference compilation", DotsCompileMode.default().enabled,
                choices=(("Compiled", True), ("Eager", False)),
            ),
        ]
    elif catalog_id == "vibevoice_local":
        result.insert(0, _checkpoint_option(VibeVoiceBaseModel.DEFAULT_REPO_ID, VibeVoiceBaseModel.PRESET_REPO_IDS))
    elif catalog_id == "moss_local":
        result.extend((
            _checkpoint_option(MossConfigs.get_default_repo_id(), tuple(config.value.repo_id for config in MossConfigs)),
            ModelOption("rolling_cont", "Rolling continuation", 0, 0,
                        MossBaseModel.ROLLING_CONTINUATION_MAX_LENGTH, integer=True,
                        info="Use a batch size of one."),
        ))
        for variant in MossConfigs:
            config = variant.value
            prefix = config.setting_prefix
            name = config.arch_name
            result.extend((
                ModelOption(f"{prefix}_temperature", f"{name} temperature", config.temperature_default,
                            config.temperature_min, config.temperature_max),
                ModelOption(f"{prefix}_top_p", f"{name} Top-P", config.audio_top_p_default,
                            config.audio_top_p_min, config.audio_top_p_max),
                ModelOption(f"{prefix}_top_k", f"{name} Top-K", config.audio_top_k_default,
                            config.audio_top_k_min, config.audio_top_k_max, integer=True),
                ModelOption(f"{prefix}_seed", f"{name} seed", None, 0, 4_294_967_295, integer=True,
                            info="Leave blank for a random seed."),
            ))
    if any(binding.name == "seed" for binding in REGISTRY.for_model(catalog_id)):
        result.append(ModelOption("seed", "Seed", None, 0, 4_294_967_295, integer=True,
                                  info="Leave blank for a random seed."))
    return tuple(result)


def validate_model_options(catalog_id: str, options: dict[str, object] | None) -> dict[str, object]:
    """Convert browser inputs to typed project setting overrides."""
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("Model settings must be an object.")
    supported = {option.key: option for option in model_options_for(catalog_id)}
    result = {}
    for key, value in options.items():
        option = supported.get(key)
        if option is None:
            raise ValueError("The selected model does not support one of these settings.")
        if value is None:
            continue
        if option.choices:
            allowed = [choice for _, choice in option.choices]
            if not any(type(value) is type(choice) and value == choice for choice in allowed):
                raise ValueError(f"Choose a supported value for {option.label}.")
        elif key == "lora_target":
            value = _validate_lora_target(value)
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Enter a number for {option.label}.")
            if option.integer:
                if int(value) != value:
                    raise ValueError(f"Enter a whole number for {option.label}.")
                value = int(value)
            if not option.minimum <= value <= option.maximum:
                raise ValueError(f"{option.label} must be between {option.minimum} and {option.maximum}.")
        if value != option.default:
            result[key] = value

    if catalog_id == _QWEN and "target" in result:
        model_type = _QWEN_TYPE_BY_SUFFIX[str(result["target"]).rsplit("-", 1)[-1]]
        if model_type != Qwen3ModelType.BASE.value and result.get("rolling_cont", 0) > 0:
            raise ValueError('Rolling continuation requires Qwen3-TTS model type "base".')
        result["model_type"] = model_type
    return result


def _validate_lora_target(value: object) -> str:
    error = "Enter an existing local adapter directory or a Hugging Face repository ID."
    if not isinstance(value, str) or len(value) > 1024 or not all(character.isprintable() for character in value):
        raise ValueError(error)
    value = value.strip()
    if not value:
        return value
    try:
        path = Path(value).expanduser()
        if path.is_dir():
            return str(path.resolve())
    except (OSError, RuntimeError):
        raise ValueError(error) from None
    if target_util.is_hf_repo_id_syntax(value) or target_util.split_hf_subfolder_target(value):
        return value
    raise ValueError(error)
