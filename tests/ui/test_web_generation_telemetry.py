import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from tts_audiobook_tool import web_models, web_render
from tts_audiobook_tool.app_types import Book, BookSection, SttVariant
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generation_events import BookRenderTelemetry, GenerationProgress
from tts_audiobook_tool.model_worker_protocol import GenerationFinished, GenerationTerminalStatus, GenerationUpdate
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_backend import BookSession


def book_session(tmp_path):
    return BookSession(Book([BookSection(phrase_groups=[
        PhraseGroup([Phrase(text, Reason.SENTENCE)]) for text in ("First line.", "Second line.")
    ])]), str(tmp_path))


def test_book_telemetry_counts_resumed_segments_and_keeps_rate_for_export(monkeypatch, tmp_path):
    updates = []
    progress = []
    missing = iter([{1}, set(), set()])
    prefs = SimpleNamespace(save_gen_log=False, stt_variant=SttVariant.DISABLED)
    state = SimpleNamespace(project=object(), prefs=prefs)
    worker_events = iter([
        GenerationUpdate("another-job", GenerationProgress(1, 0, 1, speed_multiplier=100.0)),
        GenerationUpdate("book-job", GenerationProgress(0, 1, 1)),
        GenerationUpdate("book-job", GenerationProgress(1, 0, 1, speed_multiplier=10.0)),
        GenerationFinished("book-job", GenerationTerminalStatus.COMPLETED, ""),
    ])

    def export(_state, destination, line_count, _cancel_event):
        assert line_count == 2
        assert updates[-1] == BookRenderTelemetry(2, 2, 10.0)
        return str(destination)

    monkeypatch.setattr(web_render, "_detect_model", lambda: TtsModelType.require_by_id("breeze_tts_2_local"))
    monkeypatch.setattr(web_render, "_project_state", lambda *_args, **_kwargs: nullcontext(state))
    monkeypatch.setattr(web_render, "_missing_segments", lambda *_: next(missing))
    monkeypatch.setattr(web_render, "_export_checkpoint", export)
    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", lambda **_: "book-job")
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: next(worker_events))
    monkeypatch.setattr(web_render.ModelWorker, "is_busy", lambda: False)

    output = web_render.render_book(
        book_session(tmp_path), {}, validate_audio=False, on_telemetry=updates.append,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert Path(output).name == "audiobook.abr.m4b"
    assert updates == [BookRenderTelemetry(1, 2), BookRenderTelemetry(2, 2, 10.0)]
    assert progress == [(1, 2), (1, 2), (2, 2), (2, 2)]


def test_parent_delivers_the_last_snapshot_from_a_finished_child(monkeypatch, tmp_path):
    updates = []
    progress = []

    def render(_session, _voices, *, on_telemetry, **_):
        on_telemetry(BookRenderTelemetry(1, 2))
        on_telemetry(BookRenderTelemetry(2, 2, 10.0))
        return None

    def run(command, **_):
        web_models._worker(Path(command[-1]))
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(web_render, "render_book", render)
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", lambda: None)
    monkeypatch.setattr(web_models.signal, "signal", lambda *_: None)
    monkeypatch.setattr(web_models, "_python_for", lambda _: Path(sys.executable))
    monkeypatch.setattr(web_models.subprocess, "Popen", run)

    assert web_models.render_model_book(
        "venv-breeze", book_session(tmp_path), {}, on_telemetry=updates.append,
        on_progress=lambda done, total: progress.append((done, total)),
    ) is None
    assert updates == [BookRenderTelemetry(2, 2, 10.0)]
    assert progress == [(2, 2)]
