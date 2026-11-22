"""Explicit endpoint discovery and exact remote model selection for the browser."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


REMOTE_MODEL_ID = "remote"
ROOT = Path(__file__).resolve().parents[1]
_REMOTE_ENVIRONMENT_NAMES = ("venv-remote-tts", "venv-remote", "venv-client", "venv-sgl-omni")
_REMOTE_BACKENDS = {TtsBackendKind.AUDIO_CPP, TtsBackendKind.SGL_OMNI}
_VERIFIED_EMPTY_SERVER_ISSUES = {"no_models", "no_supported_models"}


class WebRemoteTtsError(ValueError):
    """A remote TTS error suitable for display in browser controls."""


def normalize_remote_endpoint(endpoint: str | None) -> str:
    if not isinstance(endpoint, str) or not (endpoint := endpoint.strip().rstrip("/")):
        raise WebRemoteTtsError("Enter a remote TTS server URL.")
    issue = RemoteTtsDiscovery.validate_url(endpoint)
    if issue is not None:
        raise WebRemoteTtsError("Enter an HTTP or HTTPS remote TTS server URL.")
    address = urlsplit(endpoint)
    if address.username is not None or "?" in endpoint or "#" in endpoint:
        raise WebRemoteTtsError("Use the server base URL without credentials, a query, or a fragment.")
    return endpoint


def _valid_server_id(server_id: str) -> bool:
    if not isinstance(server_id, str) or not server_id.strip() or "\0" in server_id:
        return False
    try:
        server_id.encode("utf-8")
    except UnicodeError:
        return False
    return True


@dataclass(frozen=True)
class RemoteTtsSelection:
    endpoint: str
    catalog_id: str
    server_model_id: str

    def __post_init__(self):
        object.__setattr__(self, "endpoint", normalize_remote_endpoint(self.endpoint))
        if not isinstance(self.catalog_id, str) or self.backend_kind not in _REMOTE_BACKENDS:
            raise WebRemoteTtsError("Choose a supported remote speech model.")
        if not _valid_server_id(self.server_model_id):
            raise WebRemoteTtsError("Choose an exact server model from the discovered list.")

    @property
    def backend_kind(self) -> TtsBackendKind | None:
        return TtsModelType.get_by_id(self.catalog_id).value.backend_kind

    @property
    def key(self) -> str:
        identity = json.dumps([self.endpoint, self.catalog_id, self.server_model_id], ensure_ascii=False)
        return sha256(identity.encode()).hexdigest()

    @property
    def label(self) -> str:
        model = TtsModelType.get_by_id(self.catalog_id)
        return f"{model.value.ui.get('proper_name') or self.catalog_id} · {self.server_model_id}"


@dataclass(frozen=True)
class RemoteTtsConfig:
    endpoint: str = ""
    backend_kind: TtsBackendKind | None = None
    candidates: tuple[RemoteTtsSelection, ...] = ()
    selected: RemoteTtsSelection | None = None
    notice: str = ""


def remote_client_python(root: Path | str = ROOT) -> Path | None:
    """Find an existing native launcher-marked client environment in this checkout."""
    from launch import find_venvs, get_venv_python, has_remote_client_marker

    names = {name: index for index, name in enumerate(_REMOTE_ENVIRONMENT_NAMES)}
    try:
        candidates = sorted(find_venvs(str(root)), key=lambda path: names.get(Path(path).name, len(names)))
    except OSError:
        return None
    for candidate in candidates:
        interpreter = get_venv_python(candidate)
        if interpreter is None:
            continue
        try:
            if has_remote_client_marker(candidate):
                return Path(interpreter)
        except OSError:
            continue
    return None


def _candidate_selections(endpoint: str, snapshot: RemoteTtsSnapshot) -> tuple[RemoteTtsSelection, ...]:
    if snapshot.issue is not None or snapshot.backend_kind not in _REMOTE_BACKENDS:
        return ()
    ids = Counter(model["id"] for model in snapshot.models if isinstance(model.get("id"), str))
    return tuple(dict.fromkeys(
        RemoteTtsSelection(endpoint, model.id, server_id)
        for model, server_id in snapshot.candidates
        if model.value.backend_kind is snapshot.backend_kind and _valid_server_id(server_id) and ids[server_id] <= 1
    ))


def discover_remote_models(endpoint: str | None, previous: RemoteTtsConfig | None = None) -> RemoteTtsConfig:
    """Explicitly refresh one endpoint without changing process-wide native discovery state."""
    endpoint = normalize_remote_endpoint(endpoint)
    # The native forced probe accepts a URL directly. The application process may
    # serve several browser sessions, so leave its default base URL and cache alone.
    snapshot = RemoteTtsDiscovery._probe(endpoint)
    candidates = _candidate_selections(endpoint, snapshot)
    retained = previous.selected if previous is not None and previous.endpoint == endpoint else None
    selected = retained if retained in candidates else None
    if retained is None and len(candidates) == 1:
        selected = candidates[0]
    if snapshot.issue is not None:
        notice = snapshot.issue.message
    elif not candidates:
        notice = "This server has no matching remote speech models."
    elif selected is not None:
        notice = f"Selected {selected.label}."
    else:
        notice = "Choose a model from this server."
    backend = snapshot.backend_kind
    if snapshot.issue is not None and snapshot.issue.code not in _VERIFIED_EMPTY_SERVER_ISSUES:
        backend = None
    return RemoteTtsConfig(endpoint, backend, candidates, selected, notice)


def select_remote_model(endpoint: str | None, config: RemoteTtsConfig, key: str | None) -> RemoteTtsConfig:
    _check_endpoint(endpoint, config)
    if not key:
        return replace(config, selected=None, notice="Choose a model from this server.")
    candidate = next((item for item in config.candidates if item.key == key), None)
    if candidate is None:
        raise WebRemoteTtsError("Choose a model from the current server’s discovered list.")
    return replace(config, selected=candidate, notice=f"Selected {candidate.label}.")


def _check_endpoint(endpoint: str | None, config: RemoteTtsConfig) -> None:
    if not config.endpoint or normalize_remote_endpoint(endpoint) != config.endpoint:
        raise WebRemoteTtsError("Refresh remote models for the current server URL first.")


def resolve_remote_selection(endpoint: str | None, config: RemoteTtsConfig) -> RemoteTtsSelection:
    """Capture the verified pair at render submission and reject stale endpoint drafts."""
    _check_endpoint(endpoint, config)
    if (config.selected is None or config.selected not in config.candidates or config.selected.endpoint != config.endpoint
            or config.selected.backend_kind is not config.backend_kind):
        raise WebRemoteTtsError("Choose a verified model from the current remote server.")
    return config.selected


def require_exact_remote_model(selection: RemoteTtsSelection, snapshot: RemoteTtsSnapshot) -> None:
    """Confirm a previously chosen entry is still advertised at a renderer's forced probe."""
    if selection not in _candidate_selections(selection.endpoint, snapshot):
        raise WebRemoteTtsError("The selected remote model is unavailable. Refresh the server’s models and choose it again.")


@contextmanager
def remote_worker_scope(selection: RemoteTtsSelection | None) -> Iterator[None]:
    """Verify and pin the entry for the lifetime of an isolated browser worker."""
    if selection is None:
        yield
        return

    from tts_audiobook_tool.tts import Tts, WEB_REMOTE_ENTRY_ID_ENV

    Tts.init_local_model_type()
    if not Tts.is_remote_mode():
        raise WebRemoteTtsError("The selected environment is unavailable as a remote speech client.")
    previous_endpoint = RemoteTtsDiscovery.get_base_url()
    previous_entry = os.environ.get(WEB_REMOTE_ENTRY_ID_ENV)
    try:
        RemoteTtsDiscovery.set_base_url(selection.endpoint)
        require_exact_remote_model(selection, RemoteTtsDiscovery.refresh(force=True))
        os.environ[WEB_REMOTE_ENTRY_ID_ENV] = selection.server_model_id
        yield
    finally:
        RemoteTtsDiscovery.set_base_url(previous_endpoint)
        if previous_entry is None:
            os.environ.pop(WEB_REMOTE_ENTRY_ID_ENV, None)
        else:
            os.environ[WEB_REMOTE_ENTRY_ID_ENV] = previous_entry


def unload_audio_cpp_models(endpoint: str | None, config: RemoteTtsConfig) -> None:
    """Perform the operator's explicit bulk unload against the verified audio.cpp URL."""
    _check_endpoint(endpoint, config)
    if config.backend_kind is not TtsBackendKind.AUDIO_CPP:
        raise WebRemoteTtsError("Refresh an audio.cpp server before unloading its models.")
    snapshot = RemoteTtsDiscovery._probe(config.endpoint)
    if (snapshot.backend_kind is not TtsBackendKind.AUDIO_CPP
            or snapshot.issue is not None and snapshot.issue.code not in _VERIFIED_EMPTY_SERVER_ISSUES):
        raise WebRemoteTtsError("The server could not be verified as audio.cpp. Refresh remote models first.")
    message = AudioCppUtil.unload_all_models(config.endpoint)
    if message:
        raise WebRemoteTtsError(message)
