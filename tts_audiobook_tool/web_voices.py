"""Private local storage for named browser voices."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from uuid import uuid4

from filelock import FileLock, Timeout

from tts_audiobook_tool.web_render import RenderVoice


_VOICE_ID = re.compile(r"[0-9a-f]{32}")
_CLIP_NAME = re.compile(r"[0-9a-f]{64}(?:\.[a-z0-9]{1,10})?")
_FILE_SUFFIX = re.compile(r"\.[a-z0-9]{1,10}")
_INDEX_NAME = "voices.json"


class WebVoiceError(ValueError):
    """A voice library error suitable for display in the browser."""


@dataclass(frozen=True)
class VoiceSummary:
    voice_id: str
    name: str


@dataclass(frozen=True)
class SavedVoice:
    summary: VoiceSummary
    voice: RenderVoice


@dataclass(frozen=True)
class _VoiceRecord:
    voice_id: str
    voice: RenderVoice

    @property
    def summary(self) -> VoiceSummary:
        return VoiceSummary(self.voice_id, self.voice.name)


def list_voices(*, root: str | os.PathLike[str] | None = None) -> list[VoiceSummary]:
    return [record.summary for record in _read_index(_library_root(root))]


def load_voice(voice_id: str, *, root: str | os.PathLike[str] | None = None) -> SavedVoice:
    directory = _library_root(root)
    records = _read_index(directory)
    record = records[_voice_index(records, voice_id)]
    voice = record.voice
    if voice.reference_path is not None:
        voice = replace(voice, reference_path=str(_read_clip(directory / "clips", voice.reference_path)))
    return SavedVoice(record.summary, voice)


def create_voice(
        voice: RenderVoice, *, root: str | os.PathLike[str] | None = None,
        upload_root: str | os.PathLike[str] | None = None,
) -> VoiceSummary:
    with _edit_library(root, create=True) as (directory, records):
        record = _VoiceRecord(uuid4().hex, _prepare_voice(directory, voice, upload_root))
        records.append(record)
    return record.summary


def import_voices(
        voices: Iterable[RenderVoice], *, root: str | os.PathLike[str] | None = None,
        upload_root: str | os.PathLike[str] | None = None,
) -> list[VoiceSummary]:
    incoming = list(voices)
    if not incoming:
        return []

    added = []
    with _edit_library(root, create=True) as (directory, records):
        names = {record.voice.name.casefold() for record in records}
        for voice in incoming:
            name = base_name = _voice_name(voice.name)
            number = 2
            while name.casefold() in names:
                suffix = f" ({number})"
                name = base_name[:100 - len(suffix)].rstrip() + suffix
                number += 1
            names.add(name.casefold())
            voice = _prepare_voice(directory, replace(voice, name=name), upload_root)
            added.append(_VoiceRecord(uuid4().hex, voice))
        records.extend(added)
    return [record.summary for record in added]


def save_voice(
        voice_id: str, voice: RenderVoice, *, root: str | os.PathLike[str] | None = None,
        upload_root: str | os.PathLike[str] | None = None,
) -> VoiceSummary:
    with _edit_library(root) as (directory, records):
        index = _voice_index(records, voice_id)
        record = _VoiceRecord(voice_id, _prepare_voice(directory, voice, upload_root))
        records[index] = record
    return record.summary


def delete_voice(voice_id: str, *, root: str | os.PathLike[str] | None = None) -> None:
    # Open books may still refer to the immutable clip. Saved books own separate copies.
    with _edit_library(root) as (_, records):
        records.pop(_voice_index(records, voice_id))


def move_voice(voice_id: str, offset: int, *, root: str | os.PathLike[str] | None = None) -> None:
    if type(offset) is not int or offset not in (-1, 1):
        raise WebVoiceError("Move the voice up or down one position.")
    with _edit_library(root) as (_, records):
        index = _voice_index(records, voice_id)
        target = index + offset
        if 0 <= target < len(records):
            records[index], records[target] = records[target], records[index]


def _library_root(root: str | os.PathLike[str] | None, *, create: bool = False) -> Path:
    directory = Path(root) if root is not None else Path.home() / ".local/share/tts-audiobook-tool/voices"
    try:
        if create:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise OSError
        if os.name == "posix" and directory.exists():
            info = directory.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise OSError
        return directory.resolve()
    except (OSError, RuntimeError):
        raise WebVoiceError("The voice library folder must be a private directory owned by the current user.") from None


@contextmanager
def _edit_library(
        root: str | os.PathLike[str] | None, *, create: bool = False,
) -> Iterator[tuple[Path, list[_VoiceRecord]]]:
    directory = _library_root(root, create=create)
    if not directory.is_dir():
        raise WebVoiceError("The saved voice was not found.")
    lock_path = directory / ".voices.lock"
    try:
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise OSError
        with FileLock(str(lock_path), timeout=10):
            records = _read_index(directory)
            yield directory, records
            _write_index(directory, records)
    except Timeout:
        raise WebVoiceError("The voice library is busy. Try again.") from None
    except OSError:
        raise WebVoiceError("Could not save the voice library.") from None


def _voice_index(records: list[_VoiceRecord], voice_id: str) -> int:
    if not isinstance(voice_id, str) or not _VOICE_ID.fullmatch(voice_id):
        raise WebVoiceError("Choose a saved voice.")
    for index, record in enumerate(records):
        if record.voice_id == voice_id:
            return index
    raise WebVoiceError("The saved voice was not found.")


def _voice_name(name: str) -> str:
    name = name.strip()
    if not name or len(name) > 100 or len(name.splitlines()) != 1 or "\x00" in name:
        raise WebVoiceError("Enter a voice name of up to 100 characters on one line.")
    return name


def _read_index(directory: Path) -> list[_VoiceRecord]:
    index_path = directory / _INDEX_NAME
    try:
        if index_path.is_symlink():
            raise OSError
        if not index_path.exists():
            return []
        if not index_path.is_file():
            raise OSError
        with index_path.open(encoding="utf-8") as source:
            index = json.load(source)
        if type(index["version"]) is not int or index["version"] != 1 or not isinstance(index["voices"], list):
            raise ValueError

        ids = set()
        records = []
        for item in index["voices"]:
            voice_id = item["voice_id"]
            voice = RenderVoice(**item["voice"])
            if not _VOICE_ID.fullmatch(voice_id) or voice_id in ids or _voice_name(voice.name) != voice.name:
                raise ValueError
            if voice.reference_path is not None and not _CLIP_NAME.fullmatch(voice.reference_path):
                raise ValueError
            ids.add(voice_id)
            records.append(_VoiceRecord(voice_id, voice))
        return records
    except (KeyError, OSError, TypeError, ValueError):
        raise WebVoiceError("The saved voice library could not be read.") from None


def _write_index(directory: Path, records: list[_VoiceRecord]) -> None:
    payload = {"version": 1, "voices": [asdict(record) for record in records]}
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".voices-", suffix=".tmp", delete=False,
        ) as target:
            temporary = Path(target.name)
            json.dump(payload, target, indent=2, ensure_ascii=False, allow_nan=False)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, directory / _INDEX_NAME)
        temporary = None
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _prepare_voice(directory: Path, voice: RenderVoice, upload_root: str | os.PathLike[str] | None) -> RenderVoice:
    name = _voice_name(voice.name)
    if voice.reference_path is None:
        return replace(voice, name=name)
    clips = directory / "clips"
    try:
        clips.mkdir(mode=0o700, exist_ok=True)
        if clips.is_symlink() or not clips.is_dir():
            raise OSError
    except OSError:
        raise WebVoiceError("Could not open the voice library’s clips folder.") from None
    return replace(voice, name=name, reference_path=_save_clip(clips, Path(voice.reference_path), upload_root))


def _read_clip(clips: Path, name: str) -> Path:
    if not _CLIP_NAME.fullmatch(name) or clips.is_symlink():
        raise WebVoiceError("The saved voice reference clip is unavailable.")
    path = clips / name
    if path.is_symlink() or not path.is_file():
        raise WebVoiceError("The saved voice reference clip is unavailable.")
    return path


def _save_clip(clips: Path, source: Path, upload_root: str | os.PathLike[str] | None) -> str:
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        path = source.resolve(strict=True)
        if path.parent == clips:
            return _read_clip(clips, path.name).name
        if upload_root is None or not path.is_relative_to(Path(upload_root).resolve(strict=True)):
            raise OSError
    except (OSError, RuntimeError):
        raise WebVoiceError("Choose an uploaded reference clip.") from None

    suffix = path.suffix.lower()
    suffix = suffix if _FILE_SUFFIX.fullmatch(suffix) else ""
    digest = sha256()
    temporary: Path | None = None
    try:
        with path.open("rb") as uploaded, tempfile.NamedTemporaryFile(
            dir=clips, prefix=".clip-", delete=False,
        ) as target:
            temporary = Path(target.name)
            while chunk := uploaded.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())

        name = digest.hexdigest() + suffix
        destination = clips / name
        if destination.exists() or destination.is_symlink():
            _read_clip(clips, name)
        else:
            os.replace(temporary, destination)
            temporary = None
        return name
    except OSError:
        raise WebVoiceError("Could not copy the uploaded reference clip into the voice library.") from None
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()
