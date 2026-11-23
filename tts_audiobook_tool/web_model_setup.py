"""Catalog and isolated Linux setup for the browser's speech environments."""
from __future__ import annotations

from collections.abc import Callable
from ctypes.util import find_library
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import signal
import stat
import subprocess
import sys
from tempfile import mkdtemp
from threading import Event

from filelock import FileLock, Timeout

from tts_audiobook_tool.app_support.app_memory import get_nv_vram
from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_DEFAULT
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_remote_models import remote_client_python


ROOT = Path(__file__).resolve().parents[1]
INSTALL_ROOT = Path.home() / ".local/share/tts-audiobook-tool/model-envs"
_VALIDATION_RESERVE_GIB = 2
_BENCHMARK_FREE_GIB = 12
_LARGE_VARIANT_FREE_GIB = 21
_MOSS_BATCH_TARGET = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
_VIBEVOICE_LARGE_TARGET = "vibevoice/VibeVoice-7B"
_REMOTE_MARKER_REQUIREMENT = "./launcher_markers/remote_client"
_REMOTE_MARKER_PACKAGE = "tts-audiobook-tool-remote-client-marker"
_STAGED_REMOTE_MARKER = ".remote-client-marker"


class WebModelSetupError(ValueError):
    """A setup failure suitable for display in the browser."""


@dataclass(frozen=True)
class ModelSetupProgress:
    """Completed recipe steps and the stage currently being run."""

    completed: int
    total: int
    stage: str

    @property
    def description(self) -> str:
        return f"{self.stage} ({self.completed}/{self.total} steps complete)"


@dataclass(frozen=True)
class _Checkpoint:
    repo_id: str
    revision: str = ""
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    catalog_id: str | None
    python_version: str
    batch_size: int = PROJECT_BATCH_SIZE_DEFAULT
    note: str = ""
    torch_version: str = "2.8.0"
    cuda_wheels: str = "cu126"
    checkpoints: tuple[_Checkpoint, ...] = ()
    gated_repo: str | None = None
    portaudio: bool = False

    @property
    def label(self) -> str:
        if self.catalog_id is None:
            return "Remote speech client"
        return TtsModelType.require_by_id(self.catalog_id).value.ui["proper_name"]

    @property
    def requirements(self) -> str:
        if self.catalog_id is None:
            return "requirements-remote.txt"
        return TtsModelType.require_by_id(self.catalog_id).value.requirements_file_name

    @property
    def installable(self) -> bool:
        return sys.platform == "linux" and platform.machine() == "x86_64"


# Python versions and batch targets follow the README's local install and benchmark sections.
# Checkpoint names follow the adapters' default targets. Keep variant selection in model settings.
_PROFILES = (
    ModelProfile(
        "venv-breeze", "breeze_tts_2_local", "3.12", 10, "Requires NVIDIA CUDA.",
        checkpoints=(_Checkpoint("BreezeBlue/Breeze-TTS-2", "3e28c5151381a722f1d8661b4118c298caa77aa4"),),
    ),
    ModelProfile(
        "venv-chatterbox", "chatterbox_local", "3.11",
        torch_version="2.6.0", cuda_wheels="cu124",
        checkpoints=(_Checkpoint("ResembleAI/chatterbox", files=(
            "ve.pt", "t3_mtl23ls_v3.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt", "Cangjie5_TC.json",
        )),),
    ),
    ModelProfile(
        "venv-dots", "dots_local", "3.12", note="The default SOAR variant requires at least 16 GB of VRAM.",
        checkpoints=(_Checkpoint("dots-studio/dots.tts-soar"),),
    ),
    ModelProfile(
        "venv-fish-s1", "fish_s1_local", "3.12",
        note="Accept the model license on Hugging Face and run hf auth login before installing.",
        checkpoints=(_Checkpoint("fishaudio/openaudio-s1-mini"),), gated_repo="fishaudio/openaudio-s1-mini",
    ),
    ModelProfile(
        "venv-fish-s2", "fish_s2_local", "3.12",
        note="Requires 24 GB of VRAM.", checkpoints=(_Checkpoint("fishaudio/s2-pro"),),
    ),
    ModelProfile(
        "venv-glm", "glm_local", "3.11", note="Requires NVIDIA CUDA.",
        checkpoints=(_Checkpoint("zai-org/GLM-TTS"),),
    ),
    ModelProfile(
        "venv-higgs-v2", "higgs_v2_local", "3.12", note="Requires system PortAudio. 24 GB of VRAM recommended.",
        checkpoints=(
            _Checkpoint("bosonai/higgs-audio-v2-tokenizer", "9d4988fbd4ad07b4cac3a5fa462741a41810dbec"),
            _Checkpoint("bosonai/higgs-audio-v2-generation-3B-base", "10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84"),
            _Checkpoint("bosonai/hubert_base", "b4b85f1652c16ad63fdc818221b215b79ff55934"),
        ),
        portaudio=True,
    ),
    ModelProfile(
        "venv-indextts2", "indextts2_local", "3.11", checkpoints=(_Checkpoint("IndexTeam/IndexTTS-2"),),
    ),
    ModelProfile(
        "venv-mira", "mira_local", "3.12", 10, "Requires NVIDIA CUDA.",
        checkpoints=(_Checkpoint("YatharthS/MiraTTS"),),
    ),
    ModelProfile(
        "venv-moss", "moss_local", "3.12", note="The default 9B Delay variant needs 24 GB or more of VRAM.",
        torch_version="2.9.1", checkpoints=(
            _Checkpoint("OpenMOSS-Team/MOSS-TTS-v1.5", "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"),
        ),
    ),
    ModelProfile(
        "venv-omnivoice", "omnivoice_local", "3.12", checkpoints=(_Checkpoint("k2-fsa/OmniVoice"),),
    ),
    ModelProfile(
        "venv-pocket", "pocket_local", "3.12",
        note="Accept the Pocket TTS license on Hugging Face and run hf auth login for voice cloning. Language weights download on first use.",
    ),
    ModelProfile(
        "venv-qwen3tts", "qwen3tts_local", "3.12", 5, checkpoints=(_Checkpoint("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),),
    ),
    ModelProfile(
        "venv-vibevoice", "vibevoice_local", "3.11", 10, checkpoints=(_Checkpoint("microsoft/VibeVoice-1.5b"),),
    ),
)
_BY_MODEL_ID = {profile.model_id: profile for profile in _PROFILES}
_REMOTE_PROFILE = ModelProfile("venv-remote-tts", None, "3.12")


def model_profiles() -> tuple[ModelProfile, ...]:
    """Return the supported local recipes in catalog order."""
    return _PROFILES


def recommended_batch_size(
        model_id: str, *, validate_audio: bool = True, model_target: str | None = None,
) -> int:
    """Use the benchmark batch when current free VRAM can cover it and validation."""
    profile = _BY_MODEL_ID.get(model_id)
    if profile is None:
        return PROJECT_BATCH_SIZE_DEFAULT
    batch_size = profile.batch_size
    # The README's mainstream CUDA benchmarks used a 12 GiB card. The two
    # larger named variants use the separate 24 GiB benchmark as a starting point.
    needed = _BENCHMARK_FREE_GIB
    if profile.model_id == "venv-moss" and model_target == _MOSS_BATCH_TARGET:
        batch_size, needed = 20, _LARGE_VARIANT_FREE_GIB
    elif profile.model_id == "venv-vibevoice" and model_target == _VIBEVOICE_LARGE_TARGET:
        needed = _LARGE_VARIANT_FREE_GIB
    if batch_size == PROJECT_BATCH_SIZE_DEFAULT or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        return PROJECT_BATCH_SIZE_DEFAULT
    if validate_audio:
        needed += _VALIDATION_RESERVE_GIB
    memory = get_nv_vram()
    if memory is None:
        return PROJECT_BATCH_SIZE_DEFAULT
    used, total = memory
    return batch_size if (total - used) >= needed * 1024**3 else PROJECT_BATCH_SIZE_DEFAULT


def install_model(
        model_id: str, *, on_progress: Callable[[ModelSetupProgress], None] | None = None,
        cancel_event: Event | None = None,
) -> Path:
    """Install a known recipe in a private environment and publish its verified link."""
    profile = _BY_MODEL_ID.get(model_id)
    if profile is None:
        raise WebModelSetupError("Choose a supported local speech model.")
    return _install_profile(profile, on_progress, cancel_event)


def remote_client_install_supported() -> bool:
    return _REMOTE_PROFILE.installable


def install_remote_client(
    *, on_progress: Callable[[ModelSetupProgress], None] | None = None, cancel_event: Event | None = None,
) -> Path:
    """Install the native remote TTS client requirements into a launcher-marked environment."""
    return _install_profile(_REMOTE_PROFILE, on_progress, cancel_event)


def _install_profile(
    profile: ModelProfile, on_progress: Callable[[ModelSetupProgress], None] | None, cancel_event: Event | None,
) -> Path:
    if not profile.installable:
        raise WebModelSetupError("Browser installation requires an x86-64 Linux or WSL server.")
    _check_cancelled(cancel_event)

    # Both recipes create the environment, install dependencies, check the
    # dependencies, verify the environment, and publish the verified link.
    total = 5 + int(profile.catalog_id is not None) + int(profile.gated_repo is not None) + len(profile.checkpoints)
    completed = 0

    def start_step(description: str) -> None:
        nonlocal completed
        if on_progress is not None:
            on_progress(ModelSetupProgress(completed, total, description))
        # The next step starts only after this one succeeds.
        completed += 1

    if profile.catalog_id is None:
        interpreter = remote_client_python(ROOT)
        _check_cancelled(cancel_event)
        if interpreter is not None:
            if on_progress is not None:
                on_progress(ModelSetupProgress(total, total, "Verified environment already available"))
            return interpreter.parent.parent
    directory = _setup_root()
    link = ROOT / profile.model_id
    lock_path = directory / f"{profile.model_id}.lock"

    try:
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise OSError
        already_installed = False
        with FileLock(str(lock_path), timeout=0):
            if link.is_symlink() or link.exists():
                if not _qualified(link, profile):
                    raise WebModelSetupError(f"An existing path occupies {profile.model_id}. Move it before retrying setup.")
                _check_cancelled(cancel_event)
                already_installed = True
            else:
                uv = _find_uv()
                requirements = ROOT / profile.requirements
                if requirements.is_symlink() or not requirements.is_file():
                    raise WebModelSetupError(f"The upstream requirements file for {profile.label} is unavailable.")
                if profile.portaudio and not find_library("portaudio"):
                    raise WebModelSetupError("Install the PortAudio system library in WSL before setting up Higgs Audio V2.")
                if profile.gated_repo is not None:
                    _check_model_access(profile.gated_repo, start_step, cancel_event)
                _install(profile, directory, link, uv, requirements, start_step, cancel_event)
    except Timeout:
        raise WebModelSetupError(f"Setup for {profile.label} is already running.") from None
    except OSError:
        raise WebModelSetupError(f"Could not complete local setup for {profile.label}.") from None
    if on_progress is not None:
        stage = "Verified environment already available" if already_installed else "Setup complete"
        on_progress(ModelSetupProgress(total, total, stage))
    return link


def _setup_root() -> Path:
    try:
        directory = INSTALL_ROOT.resolve()
        if directory.is_relative_to(ROOT.resolve()) or directory.is_relative_to(Path("/mnt")):
            raise OSError
        INSTALL_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        if INSTALL_ROOT.is_symlink() or not INSTALL_ROOT.is_dir():
            raise OSError
        info = INSTALL_ROOT.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise OSError
        return directory
    except (OSError, RuntimeError):
        raise WebModelSetupError("The model setup directory must be a private Linux directory outside the repository.") from None


def _find_uv() -> str:
    executable = shutil.which("uv") or shutil.which(str(Path.home() / ".local/bin/uv"))
    if executable is None:
        raise WebModelSetupError("Install uv in WSL before downloading a model environment.")
    return executable


def _qualified(directory: Path, profile: ModelProfile) -> bool:
    from launch import has_remote_client_marker, probe_venv

    if not (directory / "pyvenv.cfg").is_file():
        return False
    if profile.catalog_id is None:
        return has_remote_client_marker(str(directory))
    labels, count = probe_venv(str(directory))
    return count == 1 and labels == [profile.label] and not has_remote_client_marker(str(directory))


def _install(
        profile: ModelProfile, directory: Path, link: Path, uv: str, requirements: Path,
        on_progress: Callable[[str], None] | None, cancel_event: Event | None,
) -> None:
    # Venv entry points contain absolute paths, so the verified environment remains
    # at its creation path and the checkout receives only a discovery symlink.
    stage = Path(mkdtemp(prefix=f"{profile.model_id}-", dir=directory))
    published = False
    try:
        _run_step(
            [uv, "venv", "--no-project", "--no-config", "--python", profile.python_version, str(stage)],
            f"Create the Python {profile.python_version} environment", on_progress, cancel_event,
        )
        python = str(stage / "bin" / "python")
        pip = [uv, "pip", "install", "--no-config", "--link-mode", "copy", "--python", python]
        verification = [python, "-c", _VERIFY_PYTHON_CODE, profile.python_version]
        verify_description = "Verify the Python version"
        if profile.catalog_id is not None:
            torch_index = f"https://download.pytorch.org/whl/{profile.cuda_wheels}"
            torch_build = f"{profile.torch_version}+{profile.cuda_wheels}"
            torch_pins = [f"{name}=={torch_build}" for name in ("torch", "torchaudio")]
            _run_step(
                [*pip, "--default-index", torch_index, *torch_pins],
                f"Install PyTorch {profile.torch_version} ({profile.cuda_wheels})", on_progress, cancel_event,
            )
            verification.append(torch_build)
            verify_description = "Verify the Python and CUDA wheel versions"
        normalized = stage / ".upstream-requirements.txt"
        binary_args = _normalize_requirements(requirements, normalized)
        _run_step(
            [*pip, *binary_args, "-r", str(normalized)],
            f"Install {profile.label} dependencies", on_progress, cancel_event,
        )
        normalized.unlink()
        staged_marker = stage / _STAGED_REMOTE_MARKER
        if staged_marker.is_dir():
            shutil.rmtree(staged_marker)
        _run_step([uv, "pip", "check", "--python", python], "Check the installed dependencies", on_progress, cancel_event)
        _run_step(verification, verify_description, on_progress, cancel_event)
        _check_cancelled(cancel_event)
        if not _qualified(stage, profile):
            raise WebModelSetupError(f"The new environment did not qualify as {profile.label}.")

        for checkpoint in profile.checkpoints:
            _run_step(
                [python, "-c", _SNAPSHOT_CODE, checkpoint.repo_id, checkpoint.revision, *checkpoint.files],
                f"Download {checkpoint.repo_id} weights", on_progress, cancel_event,
            )
        _check_cancelled(cancel_event)
        if on_progress is not None:
            on_progress("Publish the verified environment")
        _check_cancelled(cancel_event)
        # Creating the link directly is atomic and fails if another path appeared.
        link.symlink_to(stage, target_is_directory=True)
        published = True
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def _normalize_requirements(source: Path, destination: Path) -> list[str]:
    lines = []
    binary_args = []
    for line in source.read_text(encoding="utf-8").splitlines():
        option = line.strip().partition(" #")[0]
        if option.startswith("--only-binary="):
            for name in option.removeprefix("--only-binary=").split(","):
                binary_args.extend(("--only-binary", name.strip()))
        elif option == _REMOTE_MARKER_REQUIREMENT:
            marker = source.parent / option
            manifest = marker / "pyproject.toml"
            if marker.resolve() != marker.absolute() or manifest.is_symlink() or not manifest.is_file():
                raise WebModelSetupError("The upstream remote client launcher marker package is unavailable.")
            staged = destination.parent / _STAGED_REMOTE_MARKER
            shutil.copytree(marker, staged, ignore=shutil.ignore_patterns(".git", "build", "*.egg-info", "__pycache__"))
            lines.append(f"{_REMOTE_MARKER_PACKAGE} @ {staged.as_uri()}")
        else:
            lines.append(line)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return binary_args


_SNAPSHOT_CODE = (
    "import sys; from huggingface_hub import snapshot_download; "
    "snapshot_download(sys.argv[1], revision=sys.argv[2] or None, allow_patterns=sys.argv[3:] or None)"
)
_AUTH_CODE = "import sys; from huggingface_hub import HfApi; HfApi().auth_check(sys.argv[1])"
_VERIFY_PYTHON_CODE = (
    "import sys; from importlib.metadata import version; "
    "python_version = '.'.join(map(str, sys.version_info[:2])); "
    "raise SystemExit(0 if python_version == sys.argv[1] "
    "and (len(sys.argv) == 2 or version('torch') == version('torchaudio') == sys.argv[2]) else 1)"
)


def _check_model_access(
        repo_id: str, on_progress: Callable[[str], None] | None, cancel_event: Event | None,
) -> None:
    _run_step(
        [sys.executable, "-c", _AUTH_CODE, repo_id], f"Check Hugging Face access for {repo_id}",
        on_progress, cancel_event,
        error=f"Could not verify access to {repo_id}. Accept its license on Hugging Face and run hf auth login.",
    )


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise WebModelSetupError("Model setup was stopped.")


def _run_step(
        argv: list[str], description: str, on_progress: Callable[[str], None] | None,
        cancel_event: Event | None, *, error: str | None = None,
) -> None:
    _check_cancelled(cancel_event)
    if on_progress is not None:
        on_progress(description)
    _check_cancelled(cancel_event)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "UV_NO_PROGRESS": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    with subprocess.Popen(
        argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    ) as process:
        try:
            while True:
                _check_cancelled(cancel_event)
                try:
                    status = process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            _stop_process_group(process)
            raise
    if status != 0:
        raise WebModelSetupError(error or f"Model setup failed at '{description}' (exit {status}).")


def _stop_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
