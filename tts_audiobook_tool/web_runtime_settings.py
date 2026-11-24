"""Browser-selected runtime preferences without modifying the console preferences file."""
from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, AAC_BITRATES, APP_USER_SUBDIR, PROJECT_GEN_LOG_SUBDIR

if TYPE_CHECKING:
    from tts_audiobook_tool.prefs import Prefs


class WebRuntimeSettingsError(ValueError):
    """A browser runtime preference or availability error."""


@dataclass(frozen=True)
class WebRuntimeSettings:
    force_cpu: bool = False
    whisper_variant: str | None = None
    whisper_device: str | None = None
    save_generation_logs: bool = False
    aac_bitrate: str = AAC_BITRATE_DEFAULT
    checkpoint_format: str = "flac"

    def __post_init__(self) -> None:
        if type(self.force_cpu) is not bool or type(self.save_generation_logs) is not bool:
            raise WebRuntimeSettingsError("Choose valid speech runtime settings.")
        if self.whisper_variant is not None and SttVariant.get_by_id(self.whisper_variant) in (None, SttVariant.DISABLED):
            raise WebRuntimeSettingsError("Choose a Whisper model for validation.")
        if self.whisper_device is not None and SttConfig.from_id(self.whisper_device) is None:
            raise WebRuntimeSettingsError("Choose a supported Whisper device.")
        if self.aac_bitrate not in AAC_BITRATES:
            raise WebRuntimeSettingsError("Choose a supported AAC bitrate.")
        if self.checkpoint_format not in ("flac", "aac"):
            raise WebRuntimeSettingsError("Choose FLAC or AAC for new render checkpoints.")

    def generation_prefs(self, project_dir: str, *, validate_audio: bool) -> Prefs:
        from tts_audiobook_tool.prefs import Prefs

        variant = SttVariant.get_by_id(self.whisper_variant) if self.whisper_variant is not None else None
        device = SttConfig.from_id(self.whisper_device) if self.whisper_device is not None else None
        return Prefs(
            project_dir=project_dir,
            stt_variant=(variant or SttVariant.get_default()) if validate_audio else SttVariant.DISABLED,
            stt_config=device,
            tts_force_cpu=self.force_cpu,
            save_gen_log=self.save_generation_logs,
            aac_bitrate=self.aac_bitrate,
        )

    def checkpoint_options(self, *, validate_audio: bool) -> dict[str, object]:
        options: dict[str, object] = {}
        if self.force_cpu:
            options["force_cpu"] = True
        if validate_audio:
            if self.whisper_variant is not None and self.whisper_variant != SttVariant.get_default().id:
                options["whisper_variant"] = self.whisper_variant
            if self.whisper_device is not None:
                options["whisper_device"] = self.whisper_device
        return options

    def check_validation_availability(self, prefs: Prefs) -> None:
        """Check explicit cache and device selections at the start of synthesis."""
        if prefs.stt_variant is SttVariant.DISABLED or (self.whisper_variant is None and self.whisper_device is None):
            return

        from tts_audiobook_tool.stt import MlxWhisperAdapter, Stt

        mlx = Stt.should_use_mlx_whisper()
        if self.whisper_device is not None and prefs.stt_config is SttConfig.CUDA_FLOAT16 and not mlx:
            from tts_audiobook_tool.system_support.gpu_caps_util import GpuCapsUtil

            if not GpuCapsUtil.has_ctranslate2_float16_gpu():
                raise WebRuntimeSettingsError("CUDA float16 is unavailable for Whisper on this computer.")
        if self.whisper_variant is None:
            return

        from huggingface_hub.errors import LocalEntryNotFoundError

        variant = prefs.stt_variant.id
        try:
            if mlx:
                from huggingface_hub import snapshot_download

                snapshot = Path(snapshot_download(MlxWhisperAdapter.MODEL_MAP[variant], local_files_only=True))
                complete = (snapshot / "config.json").is_file() and any(
                    (snapshot / name).is_file() for name in ("weights.safetensors", "weights.npz")
                )
            else:
                from faster_whisper.utils import download_model

                snapshot = Path(download_model(variant, local_files_only=True))
                complete = all((snapshot / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json"))
        except LocalEntryNotFoundError as error:
            raise WebRuntimeSettingsError(f"Whisper {variant} is unavailable in the local model cache.") from error
        except (ImportError, OSError) as error:
            raise WebRuntimeSettingsError("The Whisper model cache could not be read in this speech environment.") from error
        if not complete:
            raise WebRuntimeSettingsError(f"Whisper {variant} is incomplete in the local model cache.")


def _preferences_path(path: str | os.PathLike[str] | None) -> Path:
    return Path(path) if path is not None else Path.home() / APP_USER_SUBDIR / "web-runtime-preferences.json"


def load_runtime_settings(*, path: str | os.PathLike[str] | None = None) -> WebRuntimeSettings:
    source = _preferences_path(path)
    try:
        if source.is_symlink() or source.parent.is_symlink():
            raise OSError
        if not source.exists():
            return WebRuntimeSettings()
        if not source.is_file():
            raise OSError
        payload = json.loads(source.read_text(encoding="utf-8"))
        if type(payload["version"]) is not int or payload["version"] != 1:
            raise ValueError
        return WebRuntimeSettings(**payload["runtime"])
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        raise WebRuntimeSettingsError("Could not read saved browser preferences.") from error


def save_runtime_settings(
    settings: WebRuntimeSettings, *, path: str | os.PathLike[str] | None = None,
) -> None:
    destination = _preferences_path(path)
    temporary = None
    try:
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if destination.parent.is_symlink() or destination.is_symlink():
            raise OSError
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=".browser-runtime-", delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump({"version": 1, "runtime": asdict(settings)}, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    except OSError as error:
        raise WebRuntimeSettingsError("Could not save browser preferences.") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def latest_generation_log(project_dir: str) -> str | None:
    """Find the latest browser-generated console transcript inside the active project."""
    root = Path(project_dir).resolve() / "renders"
    if root.is_symlink() or not root.is_dir():
        return None
    logs = []
    for run in root.iterdir():
        if re.fullmatch(r"[a-f0-9]{64}", run.name) is None or run.is_symlink() or not run.is_dir():
            continue
        directory = run / PROJECT_GEN_LOG_SUBDIR
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.glob("generation-*.log"):
            try:
                if path.resolve(strict=True) == path and path.is_file():
                    logs.append((path.stat().st_mtime_ns, str(path)))
            except (OSError, RuntimeError):
                continue
    return max(logs)[1] if logs else None
