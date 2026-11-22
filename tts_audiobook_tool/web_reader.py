"""Expose the bundled synchronized reader and link it to cached audio."""

from html import escape
from pathlib import Path
from urllib.parse import quote, urlencode

from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


def reader_mount() -> Mount:
    directory = Path(__file__).resolve().parents[1] / "browser_player"
    return Mount("/reader", app=StaticFiles(directory=directory, html=True))


def reader_link(cached_audio_path: str) -> str:
    audio_url = "/gradio_api/file=" + quote(cached_audio_path, safe="/")
    reader_url = "/reader/?" + urlencode({"url": audio_url})
    return f'<a href="{escape(reader_url, quote=True)}" target="_blank" rel="noopener">Open synchronized reader</a>'
