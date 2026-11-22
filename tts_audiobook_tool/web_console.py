"""Relay synthesis subprocess output to the browser as text."""

from codecs import getincrementaldecoder
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from tts_audiobook_tool.text_util import strip_ansi_codes


_LOG_NAME = "synthesis-console.log"
_READ_SIZE = 8_192
_POLL_LIMIT = 65_536
_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_PARTIAL_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*|\][^\x07\x1b]*\x1b?)?\Z")


@contextmanager
def capture_synthesis_console(
    directory: Path, on_console: Callable[[str], None] | None,
) -> Iterator[tuple[dict[str, Any], Callable[[], None]]]:
    """Yield subprocess options and a function to publish output during polling."""
    if on_console is None:
        yield {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}, lambda: None
        return

    # A separate file reader lets both subprocess streams make progress without
    # depending on polling, including while playback yields audio to the browser.
    path = directory / _LOG_NAME
    decoder = getincrementaldecoder("utf-8")(errors="replace")
    pending_escape = ""
    after_cr = False

    def emit(data: bytes, *, final: bool = False) -> None:
        nonlocal pending_escape, after_cr
        chunk = pending_escape + decoder.decode(data, final=final)
        pending_escape = ""
        if partial := _PARTIAL_ESCAPE.search(chunk):
            if not final and len(partial[0]) < _READ_SIZE:
                pending_escape = partial[0]
            chunk = chunk[:partial.start()]
        plain = _CONTROLS.sub("", strip_ansi_codes(chunk))
        if plain:
            if after_cr and plain.startswith("\n"):
                plain = plain[1:]
            after_cr = plain.endswith("\r")
            plain = plain.replace("\r\n", "\n").replace("\r", "\n")
            if plain:
                on_console(plain)

    with path.open("wb", buffering=0) as writer, path.open("rb", buffering=0) as reader:
        def drain(*, final: bool = False) -> None:
            available = os.fstat(reader.fileno()).st_size - reader.tell()
            remaining = available if final else min(available, _POLL_LIMIT)
            while remaining > 0:
                chunk = reader.read(min(_READ_SIZE, remaining))
                if not chunk:
                    break
                emit(chunk)
                remaining -= len(chunk)
            if final:
                emit(b"", final=True)

        try:
            yield {
                "stdout": writer,
                "stderr": subprocess.STDOUT,
                "env": {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            }, drain
        finally:
            drain(final=True)


def relay_synthesis_console(directory: Path) -> Callable[[str], None] | None:
    """Forward model worker events into the synthesis subprocess's output."""
    if not (directory / _LOG_NAME).is_file():
        return None

    def relay(chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()

    return relay
