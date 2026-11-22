from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from tts_audiobook_tool import web_models, web_playback, web_render
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.model_worker_protocol import ConsoleOutput, GenerationTerminalStatus, TtsPreviewFinished
from tts_audiobook_tool.web_backend import BookSession
from tts_audiobook_tool.web_console import capture_synthesis_console


def test_console_handles_split_escape_unicode_and_crlf(tmp_path):
    output = []
    with capture_synthesis_console(tmp_path, output.append) as (options, drain):
        writer = options["stdout"]
        writer.write(b"\x1b[3")
        drain()
        assert output == []
        writer.write(b"1mCaf\xc3")
        drain()
        assert output == ["Caf"]
        writer.write(b"\xa9\x1b[0m\r")
        drain()
        writer.write(b"\n\x1b]0;title")
        drain()
        writer.write(b"\x07Done\x1b[")
    assert "".join(output) == "Café\nDone"


def _substitute_worker(monkeypatch, script):
    processes = []
    popen = subprocess.Popen

    def spawn(command, **options):
        work = Path(command[-1])
        process = popen([sys.executable, "-c", script, str(work)], **options)
        processes.append((process, work))
        return process

    monkeypatch.setattr(web_models, "_python_for", lambda _model: Path(sys.executable))
    monkeypatch.setattr(web_playback, "_python_for", lambda _model: Path(sys.executable))
    monkeypatch.setattr(web_models.subprocess, "Popen", spawn)
    return processes


@pytest.mark.parametrize("action", ["book", "sample", "playback"])
def test_synthesis_console_is_live_and_drains_stdout_and_stderr(monkeypatch, tmp_path, action):
    processes = _substitute_worker(monkeypatch, """
import json
from pathlib import Path
import sys
import time

work = Path(sys.argv[1])
payload = json.loads((work / "input.json").read_text(encoding="utf-8"))
sys.stdout.write("loading\\n")
deadline = time.monotonic() + 5
while not (work / "continue").exists():
    if time.monotonic() >= deadline:
        sys.exit("console output was not delivered while the process was running")
    time.sleep(0.01)
sys.stderr.write("\\x1b[33mwarning\\x1b[0m\\n")
sys.stdout.write("x" * 200_000 + "finished")
output = None
if payload.get("action") == "sample":
    output = str(Path(payload["voice"]["name"]))
    Path(output).touch()
elif "action" not in payload:
    (work / "00000001.wav").write_bytes(b"wave")
(work / "result.json").write_text(json.dumps({"output": output}), encoding="utf-8")
""")
    output = []

    def record(chunk):
        output.append(chunk)
        if "loading\n" in chunk:
            process, work = processes[0]
            assert process.poll() is None
            (work / "continue").touch()

    if action == "sample":
        sample = tmp_path / "sample.wav"
        voice = web_render.RenderVoice(name=str(sample))
        assert web_models.render_model_sample("fake", "Hello.", voice, on_console=record) == str(sample)
    else:
        book = Book([BookSection([PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])])])
        session = BookSession(book, str(tmp_path))
        if action == "book":
            assert web_models.render_model_book("fake", session, {}, on_console=record) is None
        else:
            assert list(web_playback.play_book("fake", session, {}, first=1, last=1, on_console=record)) == [
                web_playback.PlaybackChunk(1, b"wave"),
            ]
    assert "".join(output) == "loading\nwarning\n" + "x" * 200_000 + "finished"
    assert max(map(len, output)) <= 8_192
    assert not processes[0][1].exists()


def test_stopping_a_synthesis_process_drains_its_console(monkeypatch):
    processes = _substitute_worker(monkeypatch, """
import sys
import time

sys.stdout.write("started\\n")
time.sleep(20)
""")
    output = []
    cancelled = Event()

    def record(chunk):
        output.append(chunk)
        cancelled.set()

    with pytest.raises(web_models.ModelRunError, match="Generation stopped"):
        web_models.render_model_sample(
            "fake", "Hello.", web_render.RenderVoice(), on_console=record, cancel_event=cancelled,
        )
    assert output == ["started\n"]
    assert processes[0][0].poll() is not None
    assert not processes[0][1].exists()


def test_native_worker_output_reuses_operation_scoping_and_saved_log(monkeypatch):
    terminal = TtsPreviewFinished("job", GenerationTerminalStatus.COMPLETED)
    events = iter([
        ConsoleOutput("other-job", "stdout", "unrelated"),
        ConsoleOutput("job", "stdout", "model loaded\n"),
        ConsoleOutput("job", "stderr", "model warning\n"),
        terminal,
    ])
    output = []
    saved = []
    transcript = SimpleNamespace(write_chunk=lambda chunk, stream: saved.append((chunk, stream)))
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: next(events))
    monkeypatch.setattr(web_render.ModelWorker, "is_busy", lambda: False)

    result = web_render._wait_for_job(
        "job", TtsPreviewFinished, None, transcript=transcript, on_console=output.append,
    )
    assert result is terminal
    assert output == ["model loaded\n", "model warning\n"]
    assert saved == [("model loaded\n", "stdout"), ("model warning\n", "stderr")]
