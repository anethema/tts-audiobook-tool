"""Durable working copies for the local web app, addressed by a browser-held token."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from threading import Lock
from urllib.parse import urlsplit
from uuid import uuid4
from weakref import WeakKeyDictionary

from filelock import FileLock, Timeout

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.constants_config import PROJECT_MAX_RETRIES_DEFAULT
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.web_analysis_recovery import book_key
from tts_audiobook_tool.web_backend import MAX_ANALYSIS_REQUESTS, BookSession
from tts_audiobook_tool.web_projects import (
    SavedProject, WebProjectError, _copy_asset, _copy_render_runs, _decode_snapshot, _find_project,
    _is_directory, _is_file, _project_root, _save_snapshot,
)
from tts_audiobook_tool.web_render import RenderVoice


@dataclass(frozen=True)
class SessionDraft:
    project: SavedProject
    active_project_id: str | None
    revision: str
    controls: dict
    analysis_origin: dict[str, str] | None
    omitted_request_settings: bool


# BookSession revisions use object identity. Keep only a weak reference so tab cleanup
# can release a large book. A controls-only save can then reuse its immutable book file.
_documents: WeakKeyDictionary[BookSession, tuple[Path, str]] = WeakKeyDictionary()
_documents_lock = Lock()
_ID = re.compile(r"[0-9a-f]{32}")


def draft_root() -> Path:
    return Path.home() / ".local/share/tts-audiobook-tool/web-sessions"


def _directory(token: str, root: Path | None, *, create: bool) -> Path:
    if not isinstance(token, str) or not _ID.fullmatch(token):
        raise WebProjectError("The browser session identifier is invalid.")
    parent = _project_root(root if root is not None else draft_root(), create=create)
    return _project_root(parent / token, create=create)


def bind_draft_workspace(token: str, session: BookSession, *, root: Path | None = None) -> BookSession:
    """Route new unnamed books' renders to disk before analysis or rendering starts."""
    directory = _directory(token, root, create=True)
    render_root = _project_root(directory / "render", create=True)
    current = Path(session.project_dir)
    if current.parent == render_root and re.fullmatch(r"[0-9a-f]{64}", current.name):
        return session
    render = _project_root(render_root / book_key(session.book, None), create=True)
    _copy_render_runs(Path(session.project_dir), render)
    return replace(session, project_dir=str(render))


def _safe_controls(raw: dict) -> dict:
    controls = {}
    for key in ("use_local_qwen", "regenerate_lines"):
        if type(raw.get(key)) is bool:
            controls[key] = raw[key]
    concurrency = raw.get("request_concurrency")
    if type(concurrency) in (int, float) and 1 <= concurrency <= MAX_ANALYSIS_REQUESTS and int(concurrency) == concurrency:
        controls["request_concurrency"] = int(concurrency)
    for key in ("first_line", "last_line"):
        val = raw.get(key)
        if val is None and key in raw:
            controls[key] = None
        elif type(val) in (int, float) and math.isfinite(val) and val >= 1 and int(val) == val:
            controls[key] = int(val)
    for key in ("llm_model", "reasoning_effort", "local_thinking"):
        if isinstance(raw.get(key), str):
            controls[key] = raw[key]
    if isinstance(raw.get("llm_endpoint"), str):
        endpoint = raw["llm_endpoint"].strip()
        try:
            url = urlsplit(endpoint)
            ok = url.scheme in ("https", "http") and url.hostname and url.username is None
            ok = ok and url.password is None and not url.query and not url.fragment
        except ValueError:
            ok = False
        controls["llm_endpoint"] = endpoint if ok else ""
    if isinstance(raw.get("llm_extra_params"), str):
        params = raw["llm_extra_params"].strip()
        allowed = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "frequency_penalty",
                   "presence_penalty", "seed"}
        try:
            value = json.loads(params or "{}")
            ok = isinstance(value, dict) and value.keys() <= allowed and all(
                type(v) in (int, float, bool) and math.isfinite(v) for v in value.values()
            )
        except (ValueError, OverflowError):
            ok = False
        controls["llm_extra_params"] = params if ok else ""
    return controls


@contextmanager
def _locked(directory: Path):
    try:
        with FileLock(directory / ".lock", timeout=30, mode=0o600):
            yield
    except Timeout:
        raise WebProjectError("The session is busy saving. Try again shortly.") from None
    except OSError:
        raise WebProjectError("Could not read or write the browser session.") from None


def _manifest(directory: Path) -> dict | None:
    path = directory / "session.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if not _is_file(path):
            raise ValueError
        data = json.loads(path.read_bytes())
        if (data["version"] != 1 or not _ID.fullmatch(data["revision"]) or not _ID.fullmatch(data["document"])
                or not isinstance(data["settings"], dict) or not isinstance(data["controls"], dict)):
            raise ValueError
        active = data["active_project_id"]
        if active is not None and not _ID.fullmatch(active):
            raise ValueError
        workspace = data.get("workspace")
        if workspace is not None and not re.fullmatch(r"[0-9a-f]{64}", workspace):
            raise ValueError
        return data
    except (OSError, KeyError, TypeError, ValueError):
        raise WebProjectError("The stored browser session could not be read.") from None


@lru_cache(maxsize=256)
def _cached_asset(assets: Path, source: str, stamp: tuple[int, ...]) -> str:
    return _copy_asset(assets, source)


def _draft_asset(assets: Path, source: str) -> str:
    try:
        st = Path(source).lstat()
    except OSError:
        raise WebProjectError("An uploaded session asset is no longer available.") from None
    name = _cached_asset(assets, source, (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
    if not _is_file(assets / name):
        return _copy_asset(assets, source)
    return name


def _analysis_origin(session: BookSession, book_metadata: dict) -> dict | None:
    if not session.attribution_dir or not session.attributions.ranges:
        return None
    path = Path(SpeakerAssignFileUtil.attributions_path(session.attribution_dir))
    try:
        checkpoint = json.loads(path.read_bytes())
        origin = checkpoint.get("analysis_recovery")
        if (isinstance(origin, dict) and set(origin) == {"book", "config", "baseline"}
                and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in origin.values())
                and checkpoint.get("combined_roster") == book_metadata["roster"]
                and checkpoint.get("ranges") == book_metadata["attributions"]["ranges"]):
            return origin
    except (OSError, ValueError, AttributeError):
        pass
    return None


def save_draft(
        token: str, expected_revision: str | None, session: BookSession, *, active_project_id: str | None,
        controls: dict, narrator: RenderVoice, voices: dict[str, RenderVoice], model_id: str,
        project_name: str = "", source_epub: str | None = None, validate_audio: bool = True, batch_size: int = 1,
        model_options: dict[str, dict[str, object]] | None = None, strictness: str = "",
        max_retries: int = PROJECT_MAX_RETRIES_DEFAULT, auto_concat: bool = True, performance_cues: bool = False,
        character_voices: bool = False, analyze_performance_cues: bool = False, root: Path | None = None,
) -> str:
    """Publish the snapshot and settings together. Reject writes based on an older revision."""
    if active_project_id is not None and (not isinstance(active_project_id, str) or not _ID.fullmatch(active_project_id)):
        raise WebProjectError("The active saved project identifier is invalid.")
    directory = _directory(token, root, create=True)
    with _locked(directory):
        previous = _manifest(directory)
        if expected_revision != (previous["revision"] if previous else None):
            raise WebProjectError("This session changed in another tab. Reload to open the latest saved session.")
        rev = uuid4().hex
        old_doc = previous["document"] if previous else None
        workspace = None
        if active_project_id is None:
            workspace = Path(bind_draft_workspace(token, session, root=root).project_dir).name

        def publish(book_revision: Callable[[], dict], settings: dict) -> None:
            with _documents_lock:
                reuse = old_doc is not None and _documents.get(session) == (directory, old_doc)
            if reuse and _is_file(directory / (old_doc + ".json")):
                document = old_doc
            else:
                document = uuid4().hex

                def book_payload():
                    snapshot = book_revision()
                    snapshot["analysis_origin"] = _analysis_origin(session, snapshot["web_project"])
                    return snapshot

                if JsonSaveUtil.save(JsonArtifactType.PROJECT_TEXT, directory / (document + ".json"), book_payload):
                    raise WebProjectError("Could not save the session's book.")
            safe_controls = _safe_controls(controls)
            omitted_request_settings = any(
                isinstance(controls.get(key), str) and controls[key].strip() and not safe_controls.get(key)
                for key in ("llm_endpoint", "llm_extra_params")
            )
            manifest = {
                "version": 1, "revision": rev, "document": document, "active_project_id": active_project_id,
                "workspace": workspace, "controls": safe_controls, "settings": settings,
                "omitted_request_settings": omitted_request_settings,
            }
            if JsonSaveUtil.save(JsonArtifactType.PROJECT_TEXT, directory / "session.json", lambda: manifest):
                if document != old_doc:
                    (directory / (document + ".json")).unlink(missing_ok=True)
                raise WebProjectError("Could not save the browser session.")
            with _documents_lock:
                _documents[session] = (directory, document)
            if old_doc and document != old_doc:
                with suppress(OSError):
                    (directory / (old_doc + ".json")).unlink(missing_ok=True)

        _save_snapshot(
            directory, project_name or "", session, narrator, voices, model_id, source_epub, validate_audio, batch_size,
            model_options=model_options or {}, strictness=strictness, max_retries=max_retries, auto_concat=auto_concat,
            performance_cues=performance_cues, character_voices=character_voices,
            analyze_performance_cues=analyze_performance_cues, publish_draft=publish, copy_asset=_draft_asset,
        )
        return rev


def has_draft(token: str, *, root: Path | None = None) -> bool:
    """Check the manifest without loading a second copy of the book or its assets."""
    return _manifest(_directory(token, root, create=False)) is not None


def load_draft(token: str, *, root: Path | None = None, projects_root: Path | None = None) -> SessionDraft | None:
    directory = _directory(token, root, create=False)
    if not directory.exists():
        return None
    with _locked(directory):
        manifest = _manifest(directory)
        if manifest is None:
            return None
        doc = directory / (manifest["document"] + ".json")
        try:
            if not _is_file(doc):
                raise ValueError
            snapshot = json.loads(doc.read_bytes())
            metadata = {**snapshot["web_project"], **manifest["settings"]}
        except (KeyError, TypeError, ValueError):
            raise WebProjectError("The stored session's book could not be read.") from None
        active = manifest["active_project_id"]
        if active:
            try:
                render_dir = _find_project(active, projects_root) / "render"
            except WebProjectError:
                # A deleted named project can still be reopened from this independent draft.
                active = None
        if not active:
            render_root = _project_root(directory / "render", create=True)
            render_dir = render_root / (manifest["workspace"] or manifest["document"])
        project = _decode_snapshot(directory, snapshot, metadata, render_dir)
        if active is None:
            project = replace(project, session=replace(project.session, analysis_scope=None, analysis_recovery_notice=""))
        with _documents_lock:
            _documents[project.session] = (directory, manifest["document"])
        return SessionDraft(project, active, manifest["revision"], _safe_controls(manifest["controls"]),
                            snapshot.get("analysis_origin"), manifest.get("omitted_request_settings") is True)
