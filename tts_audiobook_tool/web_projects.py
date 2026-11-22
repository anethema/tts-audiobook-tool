"""Private local storage for named browser projects."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from threading import RLock
from uuid import uuid4

from filelock import FileLock, Timeout

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.app_types.book_serialization import (
    book_from_project_text_json_dict,
    book_to_project_text_json_dict,
)
from tts_audiobook_tool.constants import PROJECT_JSON_FILE_NAME, PROJECT_TEXT_FILE_NAME
from tts_audiobook_tool.constants_config import (
    PROJECT_MAX_RETRIES_DEFAULT, PROJECT_MAX_RETRIES_MAX, PROJECT_MAX_RETRIES_MIN,
)
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.roster import Roster
from tts_audiobook_tool.system_support.platforms import is_wsl, open_directory
from tts_audiobook_tool.web_backend import BookSession, WebBackendError, new_book_session
from tts_audiobook_tool.web_render import RenderVoice


_LOCK = RLock()
_PROJECT_ID = re.compile(r"[0-9a-f]{32}")
_ASSET_NAME = re.compile(r"[0-9a-f]{64}(?:\.[a-z0-9]{1,10})?")
_FILE_SUFFIX = re.compile(r"\.[a-z0-9]{1,10}")
_RENDER_ID = re.compile(r"[0-9a-f]{64}")


class WebProjectError(ValueError):
    """A project storage error suitable for display in the browser."""


@dataclass(frozen=True)
class ProjectSummary:
    project_id: str
    name: str
    book_title: str
    updated: str


@dataclass(frozen=True)
class SavedProject:
    summary: ProjectSummary
    session: BookSession
    narrator: RenderVoice
    voices: dict[str, RenderVoice]
    model_id: str
    source_epub: str | None
    validate_audio: bool = True
    batch_size: int = 1
    model_options: dict[str, dict[str, object]] = field(default_factory=dict)
    strictness: str = ""
    max_retries: int = PROJECT_MAX_RETRIES_DEFAULT
    auto_concat: bool = True


def list_projects(*, root: str | os.PathLike[str] | None = None) -> list[ProjectSummary]:
    """Return saved projects newest first, skipping directories without a valid snapshot."""
    directory = _project_root(root)
    if not directory.exists():
        return []

    summaries = []
    try:
        for project_dir in directory.iterdir():
            if not _PROJECT_ID.fullmatch(project_dir.name) or not _is_directory(project_dir):
                continue
            try:
                _, metadata = _read_snapshot(project_dir)
                summaries.append(_summary(project_dir.name, metadata))
            except WebProjectError:
                continue
    except OSError:
        raise WebProjectError("Could not list saved projects.") from None
    return sorted(summaries, key=lambda item: item.updated, reverse=True)


def open_project_folder(project_id: str, *, root: str | os.PathLike[str] | None = None) -> Path:
    directory = _find_project(project_id, root)
    if not is_wsl():
        if error := open_directory(str(directory)):
            raise WebProjectError(error)
        return directory
    try:
        windows_path = subprocess.run(
            ["wslpath", "-w", str(directory)], check=True, text=True, capture_output=True, timeout=5,
        ).stdout.strip()
        subprocess.Popen(
            ["explorer.exe", windows_path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise WebProjectError("Could not open the project folder in Windows File Explorer.") from None
    return directory


def create_project(
        name: str,
        session: BookSession,
        *,
        narrator: RenderVoice,
        voices: dict[str, RenderVoice],
        model_id: str,
        source_epub: str | None = None,
        validate_audio: bool = True,
        batch_size: int = 1,
        model_options: dict[str, dict[str, object]] | None = None,
        strictness: str = "",
        max_retries: int = PROJECT_MAX_RETRIES_DEFAULT,
        auto_concat: bool = True,
        root: str | os.PathLike[str] | None = None,
) -> ProjectSummary:
    printable = "".join(char if char.isprintable() and char not in "/\\" else " " for char in (name or ""))
    name = " ".join(printable.split())
    if not name or len(name) > 100:
        raise WebProjectError("Enter a project name of up to 100 characters.")

    with _LOCK:
        directory = _project_root(root, create=True)
        project_dir = directory / uuid4().hex
        try:
            project_dir.mkdir(mode=0o700)
        except OSError:
            raise WebProjectError("Could not create a project directory.") from None
        try:
            return _save_snapshot(
                project_dir, name, session, narrator, voices, model_id, source_epub, validate_audio, batch_size,
                model_options={} if model_options is None else model_options,
                strictness=strictness, max_retries=max_retries, auto_concat=auto_concat,
            )
        except Exception:
            if _is_directory(project_dir):
                shutil.rmtree(project_dir, ignore_errors=True)
            raise


def save_project(
        project_id: str,
        session: BookSession,
        *,
        narrator: RenderVoice,
        voices: dict[str, RenderVoice],
        model_id: str,
        source_epub: str | None = None,
        validate_audio: bool = True,
        batch_size: int = 1,
        model_options: dict[str, dict[str, object]] | None = None,
        strictness: str | None = None,
        max_retries: int | None = None,
        auto_concat: bool | None = None,
        root: str | os.PathLike[str] | None = None,
) -> ProjectSummary:
    with _LOCK:
        project_dir = _find_project(project_id, root)
        _, previous = _read_snapshot(project_dir)
        name = _summary(project_id, previous).name
        if source_epub is None:
            source_epub = _read_asset(project_dir, previous.get("source_epub"))
        if model_options is None:
            model_options = previous.get("model_options", {})
        return _save_snapshot(
            project_dir, name, session, narrator, voices, model_id, source_epub, validate_audio, batch_size,
            model_options=model_options,
            strictness=previous.get("strictness", "") if strictness is None else strictness,
            max_retries=previous.get("max_retries", PROJECT_MAX_RETRIES_DEFAULT) if max_retries is None else max_retries,
            auto_concat=previous.get("auto_concat", True) if auto_concat is None else auto_concat,
        )


def load_project(
        project_id: str, *, root: str | os.PathLike[str] | None = None,
) -> SavedProject:
    project_dir = _find_project(project_id, root)
    snapshot, metadata = _read_snapshot(project_dir)
    summary = _summary(project_id, metadata)
    book = book_from_project_text_json_dict(snapshot)
    if isinstance(book, str):
        raise WebProjectError("The saved book text could not be read.")
    try:
        raw_roster = metadata.get("roster")
        roster = Roster.from_dict(raw_roster) if raw_roster is not None else None
        attributions = AttributionsFile.from_dict(metadata.get("attributions"))
        warnings = metadata["warnings"]
        if not isinstance(warnings, list) or not all(isinstance(warning, str) for warning in warnings):
            raise ValueError
        narrator = _read_voice(project_dir, metadata["narrator"])
        voices = {
            speaker_id: _read_voice(project_dir, voice)
            for speaker_id, voice in metadata["voices"].items()
        }
        source_epub = _read_asset(project_dir, metadata.get("source_epub"))
        model_id = metadata["model_id"]
        validate_audio = metadata.get("validate_audio", True)
        batch_size = metadata.get("batch_size", 1)
        model_options = _copy_model_options(metadata.get("model_options", {}))
        project_language = metadata.get("project_language")
        substitutions = metadata.get("word_substitutions", {})
        line_voices = {}
        for slot, voice in metadata.get("line_voices", {}).items():
            if not re.fullmatch(r"[1-9][0-9]*", slot):
                raise ValueError
            line_voices[int(slot)] = _read_voice(project_dir, voice)
        strictness = metadata.get("strictness", "")
        max_retries = metadata.get("max_retries", PROJECT_MAX_RETRIES_DEFAULT)
        auto_concat = metadata.get("auto_concat", True)
        _check_generation_settings(strictness, max_retries, auto_concat)
        if project_language is not None and not isinstance(project_language, str):
            raise ValueError
        if not isinstance(substitutions, dict) or any(
            not isinstance(source, str) or not isinstance(replacement, str)
            or not source.strip() or not replacement.strip() for source, replacement in substitutions.items()
        ):
            raise ValueError
        if not isinstance(model_id, str) or not isinstance(validate_audio, bool):
            raise ValueError
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError
    except WebProjectError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError):
        raise WebProjectError("The saved project settings could not be read.") from None

    render_dir = project_dir / "render"
    try:
        render_dir.mkdir(mode=0o700, exist_ok=True)
        if not _is_directory(render_dir):
            raise OSError
        session = new_book_session(
            book, project_dir=str(render_dir), warnings=tuple(warnings), roster=roster, attributions=attributions,
            project_language=project_language, word_substitutions=substitutions,
            line_voices=line_voices,
        )
    except (OSError, WebBackendError):
        raise WebProjectError("Could not open a workspace for this project.") from None

    return SavedProject(
        summary, session, narrator, voices, model_id, source_epub, validate_audio, batch_size,
        model_options, strictness, max_retries, auto_concat,
    )


def _check_generation_settings(strictness: str, retries: int, auto_concat: bool) -> None:
    if ((strictness != "" and Strictness.get_by_id(strictness) is None) or type(retries) is not int
            or not PROJECT_MAX_RETRIES_MIN <= retries <= PROJECT_MAX_RETRIES_MAX or type(auto_concat) is not bool):
        raise ValueError("Choose valid validation tolerance, retry count and export settings.")


def _save_snapshot(
        project_dir: Path,
        name: str,
        session: BookSession,
        narrator: RenderVoice,
        voices: dict[str, RenderVoice],
        model_id: str,
        source_epub: str | None,
        validate_audio: bool,
        batch_size: int,
        *,
        model_options: dict[str, dict[str, object]],
        strictness: str,
        max_retries: int,
        auto_concat: bool,
) -> ProjectSummary:
    try:
        model_options = _copy_model_options(model_options)
    except ValueError as error:
        raise WebProjectError("The model settings could not be saved.") from error
    try:
        _check_generation_settings(strictness, max_retries, auto_concat)
    except ValueError as error:
        raise WebProjectError(str(error)) from error
    assets = project_dir / "assets"
    try:
        assets.mkdir(mode=0o700, exist_ok=True)
        if not _is_directory(assets):
            raise OSError
    except OSError:
        raise WebProjectError("Could not open the saved project’s assets folder.") from None
    if source_epub is not None and Path(source_epub).suffix.lower() != ".epub":
        raise WebProjectError("Choose an EPUB file for the project source.")
    if not isinstance(validate_audio, bool) or type(batch_size) is not int or batch_size < 1:
        raise WebProjectError("Choose a valid audio validation setting and batch size.")

    updated = datetime.now(timezone.utc).isoformat()
    narrator_data = _save_voice(assets, narrator)
    voices_data = {speaker_id: _save_voice(assets, voice) for speaker_id, voice in voices.items()}
    line_voices_data = {str(slot): _save_voice(assets, voice) for slot, voice in session.line_voices.items()}
    source_name = _copy_asset(assets, source_epub) if source_epub else None
    summary = ProjectSummary(project_dir.name, name, session.book.title, updated)
    _copy_render_runs(Path(session.project_dir), project_dir / "render")

    def payload() -> dict:
        return {
            **book_to_project_text_json_dict(session.book),
            "web_project": {
                "version": 1,
                "name": name,
                "book_title": summary.book_title,
                "updated": updated,
                "source_epub": source_name,
                "model_id": model_id or "",
                "validate_audio": validate_audio,
                "batch_size": batch_size,
                "strictness": strictness,
                "max_retries": max_retries,
                "auto_concat": auto_concat,
                "model_options": model_options,
                "project_language": session.project_language,
                "word_substitutions": session.word_substitutions,
                "warnings": list(session.warnings),
                "roster": session.roster.to_dict() if session.roster is not None else None,
                "attributions": session.attributions.to_dict(),
                "narrator": narrator_data,
                "voices": voices_data,
                "line_voices": line_voices_data,
            },
        }

    # The private, immutable assets are ready before publishing the snapshot.
    # The standard project text reader ignores the extra browser metadata.
    if JsonSaveUtil.save(JsonArtifactType.PROJECT_TEXT, project_dir / PROJECT_TEXT_FILE_NAME, payload):
        raise WebProjectError("Could not save the project.")
    return summary


def _copy_render_runs(source_workspace: Path, destination_workspace: Path) -> None:
    if source_workspace.resolve() == destination_workspace.resolve():
        return
    source = source_workspace / "renders"
    if not _is_directory(source):
        return
    destination = destination_workspace / "renders"

    def ignored_entries(directory, names):
        ignored = {".render.lock"}
        for name in names:
            mode = (Path(directory) / name).lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                ignored.add(name)
        return ignored

    try:
        destination_workspace.mkdir(mode=0o700, exist_ok=True)
        if not _is_directory(destination_workspace):
            raise OSError
        destination.mkdir(mode=0o700, exist_ok=True)
        if not _is_directory(destination):
            raise OSError
        for run in source.iterdir():
            if not _RENDER_ID.fullmatch(run.name) or not _is_directory(run):
                continue
            target = destination / run.name
            if _is_directory(target):
                continue
            if target.exists() or target.is_symlink():
                raise OSError
            lock_path = run / ".render.lock"
            if lock_path.is_symlink():
                raise OSError
            with FileLock(str(lock_path), timeout=0):
                required = [run / PROJECT_JSON_FILE_NAME, run / PROJECT_TEXT_FILE_NAME]
                if not all(path.is_file() and not path.is_symlink() for path in required):
                    continue
                with tempfile.TemporaryDirectory(prefix=".new-run-", dir=destination) as staging:
                    copied = Path(staging) / run.name
                    shutil.copytree(run, copied, symlinks=True, ignore=ignored_entries)
                    copied.rename(target)
    except Timeout:
        raise WebProjectError("Stop the active audiobook generation before saving a copy.") from None
    except OSError:
        raise WebProjectError("Could not preserve the audiobook's generated audio.") from None


def _project_root(
        root: str | os.PathLike[str] | None, *, create: bool = False,
) -> Path:
    directory = Path(root) if root is not None else Path.home() / ".local/share/tts-audiobook-tool/web-projects"
    try:
        if create:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise OSError
        if os.name == "posix" and directory.exists():
            mode = directory.stat()
            if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077:
                raise OSError
    except OSError:
        raise WebProjectError("The saved projects folder must be a private directory owned by the current user.") from None
    return directory


def _find_project(project_id: str, root: str | os.PathLike[str] | None) -> Path:
    if not isinstance(project_id, str) or not _PROJECT_ID.fullmatch(project_id):
        raise WebProjectError("Choose a saved project.")
    project_dir = _project_root(root) / project_id
    if not _is_directory(project_dir):
        raise WebProjectError("The saved project was not found.")
    return project_dir


def _read_snapshot(project_dir: Path) -> tuple[dict, dict]:
    path = project_dir / PROJECT_TEXT_FILE_NAME
    try:
        if not _is_file(path):
            raise OSError
        with path.open(encoding="utf-8") as source:
            snapshot = json.load(source)
        metadata = snapshot["web_project"]
        if metadata["version"] != 1:
            raise ValueError
    except (KeyError, OSError, TypeError, ValueError):
        raise WebProjectError("The saved project could not be read.") from None
    return snapshot, metadata


def _summary(project_id: str, metadata: dict) -> ProjectSummary:
    try:
        name = metadata["name"]
        book_title = metadata["book_title"]
        updated = metadata["updated"]
        if not all(isinstance(value, str) for value in (name, book_title, updated)):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise WebProjectError("The saved project details could not be read.") from None
    return ProjectSummary(project_id, name, book_title, updated)


def _copy_model_options(data: object) -> dict[str, dict[str, object]]:
    """Snapshot settings by environment. Available model catalogs may change."""
    if not isinstance(data, dict):
        raise ValueError
    options = {}
    for model_id, settings in data.items():
        if not isinstance(model_id, str) or not isinstance(settings, dict):
            raise ValueError
        for name, value in settings.items():
            if not isinstance(name, str) or type(value) not in (str, int, float, bool, type(None)):
                raise ValueError
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError
        options[model_id] = dict(settings)
    return options


def _save_voice(assets: Path, voice: RenderVoice) -> dict:
    data = asdict(voice)
    reference_path = data.pop("reference_path")
    data["reference"] = _copy_asset(assets, reference_path) if reference_path else None
    return data


def _read_voice(project_dir: Path, data: dict) -> RenderVoice:
    if not isinstance(data, dict):
        raise ValueError
    settings = {
        field.name: data[field.name] for field in fields(RenderVoice)
        if field.name != "reference_path" and field.name in data
    }
    return RenderVoice(_read_asset(project_dir, data["reference"]), **settings)


def _read_asset(project_dir: Path, name: str | None) -> str | None:
    if name is None:
        return None
    assets = project_dir / "assets"
    if not isinstance(name, str) or not _ASSET_NAME.fullmatch(name) or not _is_directory(assets):
        raise WebProjectError("A saved project file is unavailable.")
    asset = assets / name
    if not _is_file(asset):
        raise WebProjectError("A saved project file is unavailable.")
    return str(asset)


def _copy_asset(assets: Path, source: str) -> str:
    path = Path(source)
    temporary: Path | None = None
    try:
        if not _is_file(path):
            raise OSError
        if path.parent.resolve() == assets.resolve() and _ASSET_NAME.fullmatch(path.name):
            return path.name
        suffix = path.suffix.lower()
        suffix = suffix if _FILE_SUFFIX.fullmatch(suffix) else ""
        digest = sha256()
        with path.open("rb") as uploaded, tempfile.NamedTemporaryFile(
            dir=assets, prefix=".upload-", delete=False,
        ) as target:
            temporary = Path(target.name)
            while chunk := uploaded.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())

        name = digest.hexdigest() + suffix
        destination = assets / name
        if not _is_file(destination):
            os.replace(temporary, destination)
            temporary = None
        return name
    except OSError:
        raise WebProjectError("Could not copy an uploaded file into the saved project.") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _is_directory(path: Path) -> bool:
    return not path.is_symlink() and path.is_dir()


def _is_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()
