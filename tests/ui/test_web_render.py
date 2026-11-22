import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from tts_audiobook_tool import web_render
from tts_audiobook_tool.app_types import Book, BookSection, Sound
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.model_worker_protocol import (
    GenerationFinished,
    GenerationTerminalStatus,
    TtsPreviewFinished,
)
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def test_sample_returns_wav_and_removes_worker_project(monkeypatch):
    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render, "mark_interactive_main", lambda: None)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    scratch_paths = []

    def submit(*, state, prompt):
        assert prompt == "An evening in the garden."
        assert state.project.voice_references[0].instructions == {model.id: "Read gently."}
        project_dir = Path(state.project.dir_path)
        assert (project_dir / "project.json").is_file()
        scratch_paths.append(project_dir)
        return "job"

    sample = Sound(np.full(300, 0.125, dtype=np.float32), 24000)
    result = TtsPreviewFinished("job", GenerationTerminalStatus.COMPLETED, sample)
    monkeypatch.setattr(web_render.ModelWorker, "submit_tts_preview", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: result)
    output = Path(web_render.render_sample(" An evening in the garden. ", web_render.RenderVoice(
        instructions="Read gently.",
    )))
    try:
        data, sr = soundfile.read(output)
        assert sr == sample.sr
        assert len(data) == len(sample.data)
        assert not scratch_paths[0].exists()
    finally:
        output.unlink()


def test_cancelled_stuck_worker_is_shut_down(monkeypatch):
    calls = []
    cancel_event = threading.Event()
    cancel_event.set()
    monkeypatch.setattr(web_render, "_CANCEL_GRACE_SECONDS", 0)
    monkeypatch.setattr(web_render.ModelWorker, "request_cancel", lambda op: calls.append(("cancel", op)))
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: None)
    monkeypatch.setattr(web_render.ModelWorker, "is_busy", lambda: True)
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", lambda: calls.append(("shutdown",)))
    with pytest.raises(web_render.WebRenderError, match="stopped"):
        web_render._wait_for_job("job", TtsPreviewFinished, cancel_event)
    assert calls == [("cancel", "job"), ("shutdown",)]


def test_cast_prefers_named_narrator_and_preserves_original_book(monkeypatch):
    book = Book([BookSection(phrase_groups=[
        PhraseGroup(phrases=[Phrase(text=text, reason=Reason.SENTENCE)]) for text in ("Opening", "Hello", "Again", "Yes")
    ])])
    attributions = AttributionsFile()
    monkeypatch.setattr(
        AttributionsFile, "speaker_of_each_line",
        lambda _self, _book: {1: "ALEX", 2: "ALEX", 3: "UNKNOWN"},
    )
    session = SimpleNamespace(
        book=book,
        roster=SimpleNamespace(narrator=SimpleNamespace(id="NINA")),
        attributions=attributions,
    )
    nina = web_render.RenderVoice(instructions="Calm")
    alex = web_render.RenderVoice(instructions="Quiet")
    cast_book, voices = web_render._cast_book(
        session, {"NINA": nina, "ALEX": alex, "NARRATOR": web_render.RenderVoice()},
    )
    assert voices == [nina, alex]
    assert [group.voice_index for group in cast_book.phrase_groups] == [0, 1, 1, 0]
    assert [group.voice_index for group in book.phrase_groups] == [0] * 4


def test_process_exit_reaps_idle_model_worker(tmp_path):
    probe = tmp_path / "worker_exit_probe.py"
    probe.write_text(
        "from tts_audiobook_tool.web_render import ModelWorker\n"
        "if __name__ == '__main__':\n"
        "    assert ModelWorker.start() == ''\n"
        "    assert ModelWorker.is_alive()\n"
        "    print('worker started', flush=True)\n",
        encoding="utf-8",
    )
    repo = str(Path(web_render.__file__).parent.parent)
    paths = [repo, os.environ.get("PYTHONPATH", "")]
    result = subprocess.run(
        [sys.executable, str(probe)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "worker started" in result.stdout



def test_book_retries_only_missing_audio_and_keeps_completed_export(monkeypatch, tmp_path):
    from tts_audiobook_tool.app_types.validation_result import SkippedResult
    from tts_audiobook_tool.generate_util import GenerateUtil

    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render, "mark_interactive_main", lambda: None)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    book = Book([BookSection(phrase_groups=[
        PhraseGroup(phrases=[Phrase(text=text, reason=Reason.SENTENCE)])
        for text in ("An evening in the garden.", "A fox walked by.")
    ])])
    session = SimpleNamespace(
        book=book, roster=None, attributions=AttributionsFile(), project_dir=str(tmp_path),
    )
    submissions = []
    statuses = iter([
        GenerationTerminalStatus.CANCELLED,
        GenerationTerminalStatus.COMPLETED,
        GenerationTerminalStatus.CANCELLED,
    ])
    time = np.arange(24000, dtype=np.float32) / 24000
    sample = Sound(0.12 * np.sin(2 * np.pi * time * 440), 24000)

    def submit(*, state, indices, batch_size, is_regen):
        submissions.append(indices)
        index = min(indices)
        error, _ = GenerateUtil.save_sound_and_timing_json(
            state, state.project.phrase_groups[index], index, SkippedResult(sample, "Validation skipped"),
            is_real_time=False,
        )
        assert not error
        return "job"

    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", submit)
    monkeypatch.setattr(
        web_render.ModelWorker, "get_event",
        lambda **_: GenerationFinished("job", next(statuses), ""),
    )
    with pytest.raises(web_render.WebRenderError, match="stopped"):
        web_render.render_book(session, {}, validate_audio=False)
    output = web_render.render_book(session, {}, validate_audio=False)
    assert submissions == [{0, 1}, {1}]
    assert Path(output).suffix == ".m4b"
    assert Path(output).stat().st_size > 0
    assert web_render.render_book(session, {}, validate_audio=False) == output
    assert submissions == [{0, 1}, {1}]

    updated_voices = {"NARRATOR": web_render.RenderVoice(instructions="Read slowly.")}
    with pytest.raises(web_render.WebRenderError, match="stopped"):
        web_render.render_book(session, updated_voices, validate_audio=False)
    assert submissions[-1] == {0, 1}

def test_reference_transcription_uses_cached_cpu_model(monkeypatch, tmp_path):
    from faster_whisper import utils
    from tts_audiobook_tool import stt
    from tts_audiobook_tool.app_types import ConcreteSegment, ConcreteWord, SttVariant

    clip = tmp_path / "reference.wav"
    time = np.arange(16000, dtype=np.float32) / 16000
    soundfile.write(clip, 0.12 * np.sin(2 * np.pi * time * 440), 16000)
    for file_name in ("model.bin", "tokenizer.json", "config.json"):
        (tmp_path / file_name).touch()

    def lookup(model_name, *, local_files_only):
        assert model_name == SttVariant.get_default().id
        assert local_files_only
        return str(tmp_path)

    class FakeWhisper:
        supported_languages = ["en"]

        def __init__(self, path, *, device, compute_type, cpu_threads):
            assert Path(path) == tmp_path
            assert device == "cpu"
            assert compute_type == "int8_float32"
            assert cpu_threads > 0

        def transcribe(self, data, *, language, word_timestamps):
            assert len(data) == 16000
            assert language == "en"
            assert word_timestamps
            return [ConcreteSegment(0.0, 0.3, "A clear word.", [
                ConcreteWord(0.0, 0.2, " Clear", 0.9),
                ConcreteWord(0.2, 0.3, " faint", 0.1),
            ])], None

    monkeypatch.setattr(utils, "download_model", lookup)
    monkeypatch.setattr(stt, "FasterWhisperAdapter", FakeWhisper)
    monkeypatch.setattr(web_render, "mark_interactive_main", lambda: None)
    assert web_render.transcribe_reference(str(clip), language_code="en-US") == "Clear"
