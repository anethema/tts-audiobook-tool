"""Private checkpoints for reopening the same book and dialogue analysis configuration."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

from filelock import FileLock, Timeout

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.app_types.book_serialization import book_to_json_dict
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.roster import Roster


class AnalysisRecoveryError(ValueError):
    """A recovery error that can be displayed in the browser."""


def _digest(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(data.encode("utf-8")).hexdigest()


def recovery_root() -> Path:
    return Path.home() / ".local/share/tts-audiobook-tool/analysis-checkpoints"


def book_key(book: Book, scope: str | None) -> str:
    return _digest((scope, book_to_json_dict(book)))


def baseline_key(roster: Roster, attributions: AttributionsFile) -> str:
    return _digest((roster.to_dict(), attributions.to_dict()))


def settings_key(provider, performance_cues: bool, character_voices: bool) -> str:
    # Credentials are unnecessary for identifying equivalent analysis requests.
    return _digest((provider, performance_cues, character_voices))


def checkpoint_dir(book: str, baseline: str) -> Path:
    return recovery_root() / _digest((book, baseline))


def _private_directory(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink() and (
        os.name != "posix" or (path.stat().st_uid == os.getuid() and not path.stat().st_mode & 0o077)
    )


def available_notice(directory: Path) -> str:
    """Report the largest saved result for this initial state across analysis settings."""
    if not (_private_directory(directory.parent) and _private_directory(directory)):
        return ""
    counts = []
    for path in directory.glob("*.json"):
        try:
            value = json.loads(path.read_bytes())
            counts.append(len(value["ranges"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not counts:
        return ""
    count = max(counts)
    return (f"Dialogue recovery available with {count:,} saved chunk range{'s' if count != 1 else ''}. "
            "Use the matching analysis settings to resume.")


@contextmanager
def locked_checkpoint(directory: Path, config: str):
    root = directory.parent
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.mkdir(mode=0o700, exist_ok=True)
        if not (_private_directory(root) and _private_directory(directory)):
            raise OSError
        # Do not hold a second paid request open while a matching run owns this checkpoint.
        lock = FileLock(directory / (config + ".lock"), timeout=0, mode=0o600)
        lock.acquire()
    except Timeout:
        raise AnalysisRecoveryError("Another analysis is already working on this book with these settings.") from None
    except OSError:
        raise AnalysisRecoveryError("Could not open the private dialogue recovery folder.") from None
    try:
        yield directory / (config + ".json")
    finally:
        lock.release()


def copy_checkpoint(source: Path, destination: Path) -> None:
    """Publish a whole, already-serialized checkpoint with a private atomic replacement."""
    temporary = None
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with source.open("rb") as src, NamedTemporaryFile(dir=destination.parent, delete=False) as dst:
            temporary = Path(dst.name)
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
    except OSError:
        raise AnalysisRecoveryError("Dialogue analysis progress could not be copied to its workspace.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
