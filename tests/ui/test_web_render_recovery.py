from pathlib import Path
from threading import Event

import numpy as np
import pytest

from tts_audiobook_tool import web_render
from tts_audiobook_tool.app_types import Book, BookSection, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.validation_result import SkippedResult
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.generation_events import BookRenderTelemetry, GenerationProgress, GenerationTimedOut, ModelUnhealthy
from tts_audiobook_tool.model_worker_protocol import (
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    WorkerCommandFailed,
    WorkerExited,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_backend import BookSession
from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettings, latest_generation_log


@pytest.fixture
def session(monkeypatch, tmp_path):
    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render, "mark_interactive_main", lambda: None)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    monkeypatch.setattr(web_render.ModelWorker, "is_busy", lambda: False)
    monkeypatch.setattr(web_render.ModelWorker, "drain_events", lambda: [])
    return BookSession(Book([BookSection(phrase_groups=[
        PhraseGroup([Phrase(text, Reason.SENTENCE)]) for text in ("First.", "Second.", "Third.", "Fourth.")
    ])]), str(tmp_path))


def _save_take(state, index, staging_dir="") -> Path:
    time = np.arange(2400, dtype=np.float32) / 24000
    sound = Sound(0.12 * np.sin(2 * np.pi * time * 440), 24000)
    error, filename = GenerateUtil.save_sound_and_timing_json(
        state, state.project.phrase_groups[index], index, SkippedResult(sound, "Mock audio"),
        is_real_time=False, dir_path=staging_dir or None,
    )
    assert not error
    return Path(filename)


def test_timeout_rescans_after_worker_stops_and_assembles_once(monkeypatch, session):
    requests = []
    pending_saves = []
    exports = []
    telemetry = []
    console = []
    checkpoint_dirs = []

    def submit(*, state, indices, batch_size, guard_first_batch, **_):
        requests.append((indices, batch_size, guard_first_batch))
        checkpoint_dirs.append(state.project.dir_path)
        if len(requests) == 2:
            pending_saves.append((state, 1))
        else:
            for index in indices:
                _save_take(state, index)
        return f"job-{len(requests)}"

    def shutdown():
        state, index = pending_saves.pop()
        _save_take(state, index)

    def export(_state, output, total, _cancel_event):
        assert total == 4
        exports.append(output)
        return str(output)

    events = iter([
        GenerationFinished("job-1", GenerationTerminalStatus.COMPLETED, ""),
        GenerationUpdate("job-2", GenerationProgress(2, 1, 3, (2, 3), eta_seconds=50, saved=0)),
        GenerationUpdate("job-2", GenerationTimedOut(180)),
        GenerationFinished("job-2", GenerationTerminalStatus.ABORTED, ""),
        GenerationFinished("job-3", GenerationTerminalStatus.COMPLETED, ""),
    ])
    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: next(events))
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", shutdown)
    monkeypatch.setattr(web_render, "_export_checkpoint", export)
    render_settings = dict(validate_audio=False, batch_size=8)

    assert web_render.render_book(session, {}, last_line=1, auto_concat=False, **render_settings) is None
    output = web_render.render_book(
        session, {}, on_telemetry=telemetry.append, on_console=console.append,
        runtime_settings=WebRuntimeSettings(save_generation_logs=True), **render_settings,
    )

    assert requests == [({0}, 8, False), ({1, 2, 3}, 8, True), ({2, 3}, 4, True)]
    assert not pending_saves
    assert len(set(checkpoint_dirs)) == 1
    assert exports == [Path(output)]
    assert telemetry == [
        BookRenderTelemetry(1, 4),
        BookRenderTelemetry(1, 4, eta_seconds=50),
        BookRenderTelemetry(2, 4),
        BookRenderTelemetry(4, 4),
        BookRenderTelemetry(4, 4, assembling=True),
    ]
    assert "Last attempted lines 3-4" in "".join(console)
    assert "batch size 4" in Path(latest_generation_log(session.project_dir)).read_text(encoding="utf-8")


def test_retry_limit_resets_only_when_more_audio_is_saved(monkeypatch, session):
    requests = []
    shutdowns = []
    console = []

    def submit(*, state, indices, batch_size, guard_first_batch, **_):
        requests.append((indices, batch_size, guard_first_batch))
        if len(requests) == 3:
            _save_take(state, 1)
        return f"job-{len(requests)}"

    events = iter([
        GenerationUpdate("job-1", ModelUnhealthy("TTS model ran out of memory")),
        WorkerExited("job-2", "Worker exited unexpectedly."),
        GenerationUpdate("job-3", ModelUnhealthy("TTS model ran out of memory")),
        GenerationUpdate("job-4", GenerationTimedOut(180)),
        GenerationUpdate("job-5", GenerationProgress(0, 1, 1, (2,), saved=0)),
        WorkerExited("job-5", "Worker exited unexpectedly."),
    ])
    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: next(events))
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", lambda: shutdowns.append(len(requests)))
    monkeypatch.setattr(web_render, "_export_checkpoint", lambda *_: pytest.fail("Incomplete audio was exported"))

    with pytest.raises(web_render.WebRenderError, match=r"Last attempted lines 3.*retries without saving more audio"):
        web_render.render_book(session, {}, first_line=2, last_line=3, batch_size=8, validate_audio=False,
                               on_console=console.append)

    assert requests == [
        ({1, 2}, 8, False), ({1, 2}, 4, True), ({1, 2}, 2, True), ({2}, 1, True), ({2}, 1, True),
    ]
    assert shutdowns == [1, 2, 3, 4, 5]
    assert "1 selected line still needs audio (3)." in console[-1]
    assert len(list(Path(session.project_dir).glob("renders/*/segments/*.flac"))) == 1


def test_regeneration_resumes_complete_staged_takes_and_keeps_the_old_takes_until_publication(monkeypatch, session):
    requests = []
    old_takes = []

    def submit(*, state, indices, is_regen, staging_dir, **_):
        requests.append((indices, staging_dir))
        if not is_regen:
            for index in indices:
                old_takes.append(_save_take(state, index))
        elif len(requests) == 2:
            _save_take(state, 1, staging_dir)
            _save_take(state, 2, staging_dir).write_bytes(b"truncated FLAC")
        else:
            assert all(path.is_file() for path in old_takes)
            assert len(list(Path(staging_dir).glob("*.flac"))) == 1
            _save_take(state, 2, staging_dir)
        return f"job-{len(requests)}"

    events = iter([
        GenerationFinished("job-1", GenerationTerminalStatus.COMPLETED, ""),
        GenerationUpdate("job-2", GenerationTimedOut(180)),
        GenerationFinished("job-3", GenerationTerminalStatus.COMPLETED, ""),
    ])
    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: next(events))
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", lambda: None)

    settings = dict(batch_size=4, validate_audio=False, auto_concat=False)
    web_render.render_book(session, {}, **settings)
    web_render.render_book(session, {}, first_line=2, last_line=3, regenerate=True, **settings)

    staging_dir = requests[1][1]
    assert requests == [({0, 1, 2, 3}, ""), ({1, 2}, staging_dir), ({2}, staging_dir)]
    assert staging_dir and not Path(staging_dir).exists()
    published = sorted(path for path in old_takes[0].parent.glob("*.flac") if " [take " in path.stem)
    assert [path.name[:7] for path in published] == ["[00002]", "[00003]"]


@pytest.mark.parametrize(("terminal", "cancel", "error"), [
    (WorkerCommandFailed("job", "Invalid worker command"), False, "worker stopped"),
    (GenerationFinished("job", GenerationTerminalStatus.FAILED, "", "CheckpointWriteError: Storage full"),
     False, "Storage full"),
    (GenerationUpdate("job", GenerationTimedOut(180)), True, "Rendering stopped"),
    (GenerationUpdate("job", GenerationTimedOut(180, is_remote=True)), False, "audiobook batch exceeded"),
])
def test_permanent_cancelled_and_remote_failures_do_not_resubmit(monkeypatch, session, terminal, cancel, error):
    submissions = []
    cancel_event = Event()

    def get_event(**_):
        if cancel:
            cancel_event.set()
        return terminal

    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", lambda **_: submissions.append("job") or "job")
    monkeypatch.setattr(web_render.ModelWorker, "get_event", get_event)
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", lambda: None)

    with pytest.raises(web_render.WebRenderError, match=error):
        web_render.render_book(session, {}, validate_audio=False, cancel_event=cancel_event)

    assert submissions == ["job"]
