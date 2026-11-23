from dataclasses import asdict, replace
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from tts_audiobook_tool import model_runtime, web_models, web_render
from tts_audiobook_tool.app_types import Book, BookSection, RoomReverb, Sound, Strictness, SttConfig, SttVariant
from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, AAC_BITRATES
from tts_audiobook_tool.generation_events import GenerationPhase
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleOutput,
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    TtsPreviewFinished,
)
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_backend import BookSession


@pytest.fixture(autouse=True)
def preserve_runtime_role(monkeypatch):
    # The renderer normally marks its own subprocess as the interactive owner.
    monkeypatch.setattr(model_runtime, "_role", model_runtime.current_role())
from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettings, latest_generation_log


def test_sample_returns_wav_and_removes_worker_project(monkeypatch):
    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render, "mark_interactive_main", lambda: None)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    scratch_paths = []

    def submit(*, state, prompt):
        assert prompt == "An evening in the garden."
        assert state.project.voice_references[0].instructions == {model.id: "Read gently."}
        assert state.project.get_model_setting(model.id, "temperature") == 0.65
        project_dir = Path(state.project.dir_path)
        saved = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        assert saved["model_settings"]["models"][model.id]["parameters"]["temperature"] == 0.65
        scratch_paths.append(project_dir)
        return "job"

    sample = Sound(np.full(300, 0.125, dtype=np.float32), 24000)
    result = TtsPreviewFinished("job", GenerationTerminalStatus.COMPLETED, sample)
    monkeypatch.setattr(web_render.ModelWorker, "submit_tts_preview", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: result)
    output = Path(web_render.render_sample(
        " An evening in the garden. ", web_render.RenderVoice(instructions="Read gently."),
        model_options={"temperature": 0.65},
    ))
    try:
        data, sr = soundfile.read(output)
        assert sr == sample.sr
        assert len(data) == len(sample.data)
        assert not scratch_paths[0].exists()
    finally:
        output.unlink()


def test_sample_rejects_settings_outside_selected_model_options(monkeypatch):
    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    with pytest.raises(web_render.WebRenderError):
        web_render.render_sample("A preview", web_render.RenderVoice(), model_options={"file_name": ["clip.wav"]})


def test_force_cpu_rejects_cuda_only_model_with_native_message(monkeypatch):
    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render.Tts, "_force_cpu", False)
    monkeypatch.setattr(web_render.ModelWorker, "submit_tts_preview", lambda **_: pytest.fail("Preview started"))
    with pytest.raises(web_render.WebRenderError, match="Breeze TTS 2 does not support CPU inference"):
        web_render.render_sample(
            "Preview", web_render.RenderVoice(), runtime_settings=WebRuntimeSettings(
                force_cpu=True, whisper_device=SttConfig.CPU_INT8FLOAT32.id,
            ),
        )


def test_sample_uses_trim_model_instructions_and_voice_effects(monkeypatch, tmp_path):
    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    rendered = Sound(np.full(2400, 0.2, dtype=np.float32), 24000)
    clip = tmp_path / "reference.wav"
    time = np.arange(24000 * 4, dtype=np.float32) / 24000
    source = 0.2 * np.sin(2 * np.pi * 440 * time)
    source[36000:60000] = 0
    soundfile.write(clip, source, 24000)

    def submit(*, state, prompt):
        voice = state.project.voice_references[0]
        assert voice.instructions[model.id] == "Read with urgency."
        assert voice.transcript == "The selected words."
        assert voice.name == "Storyteller"
        assert voice.room_reverb is RoomReverb.SMALL_DRR22
        reference = Path(state.project.dir_path) / "voice" / voice.file_name
        assert soundfile.info(reference).duration == pytest.approx(2.25, abs=0.03)
        return "job"

    monkeypatch.setattr(web_render.ModelWorker, "submit_tts_preview", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: TtsPreviewFinished(
        "job", GenerationTerminalStatus.COMPLETED, rendered,
    ))
    settings = web_render.RenderVoice(
        str(clip), "The selected words.", "Read gently.", name="Storyteller",
        trim_start=0.5, trim_end=3.5, pan=-1, volume=50, room_reverb=RoomReverb.SMALL_DRR22.id,
        model_instructions=((model.id, " Read with urgency. "),),
        max_reference_silence=0.25,
    )
    output = Path(web_render.render_sample("Preview line", settings))
    try:
        data, sr = soundfile.read(output)
        assert sr == rendered.sr
        assert data.shape[1] == 2
        assert data[0] == pytest.approx([0.1 * np.sqrt(2), 0], abs=0.0001)
        assert len(data) > len(rendered.data)
        assert np.abs(data[len(rendered.data):]).max() > 0
    finally:
        output.unlink()


def test_qwen_custom_voice_reaches_native_worker_and_takes_precedence_over_clip(monkeypatch, tmp_path):
    from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel

    model = TtsModelType.require_by_id("qwen3tts_local")
    target = next(target for target in Qwen3BaseModel.PRESET_REPO_IDS if target.endswith("-CustomVoice"))
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    settings = web_render.RenderVoice(
        reference_path=str(tmp_path / "unused.wav"), presets=((model.id, "Vivian"), ("pocket_local", "alba")),
    )

    def submit(*, state, prompt):
        assert state.project.get_model_setting(model.id, "model_type") == "custom_voice"
        voice = state.project.voice_references[0]
        assert voice.presets == {model.id: "vivian", "pocket_local": "alba"}
        assert not voice.file_name
        return "job"

    rendered = Sound(np.full(240, 0.2, dtype=np.float32), 24000)
    monkeypatch.setattr(web_render.ModelWorker, "submit_tts_preview", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: TtsPreviewFinished(
        "job", GenerationTerminalStatus.COMPLETED, rendered,
    ))
    output = Path(web_render.render_sample("Preview line", settings, model_options={"target": target}))
    try:
        assert soundfile.info(output).frames == rendered.data.size
    finally:
        output.unlink()


def test_pocket_preset_uses_no_reference_audio_in_render_project_or_checkpoint(tmp_path):
    from tts_audiobook_tool.project import Project

    model = "pocket_local"
    voice = web_render.RenderVoice(reference_path=str(tmp_path / "unavailable.wav"), presets=((model, "Alba"),))
    native = web_render._build_voice(Project(tts_model_type=model), 0, voice)
    assert native.presets == {model: "alba"}
    assert not native.file_name
    without_clip = replace(voice, reference_path=None)
    assert web_render._checkpoint_id(Book([]), [voice], model, "en", False, 1) == web_render._checkpoint_id(
        Book([]), [without_clip], model, "en", False, 1,
    )


def test_pocket_sample_applies_the_browser_model_default(monkeypatch):
    model = TtsModelType.require_by_id("pocket_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render, "mark_interactive_main", lambda: None)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))

    def submit(*, state, prompt):
        assert state.project.get_model_setting(model.id, "model_code") == "english_2026-10"
        return "job"

    sample = Sound(np.full(300, 0.125, dtype=np.float32), 24000)
    result = TtsPreviewFinished("job", GenerationTerminalStatus.COMPLETED, sample)
    monkeypatch.setattr(web_render.ModelWorker, "submit_tts_preview", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: result)
    voice = web_render.RenderVoice(presets=((model.id, "Alba"),))
    output = Path(web_render.render_sample("A line.", voice))
    try:
        assert soundfile.info(output).frames == sample.data.size
    finally:
        output.unlink()


def test_pocket_default_preserves_existing_explicit_checkpoint_id():
    model = "pocket_local"
    args = (Book([]), [web_render.RenderVoice()], model, "en", False, 1)
    explicit = {"model_code": "english_2026-10"}
    existing = web_render._checkpoint_id(*args, model_options=explicit)
    for option in (None, {}, {"model_code": ""}, explicit):
        effective = web_render._validated_model_options(model, option)
        assert effective == explicit
        assert web_render._checkpoint_id(*args, model_options=effective) == existing
    alternative = {"model_code": "french_24l"}
    assert web_render._validated_model_options(model, alternative) == alternative
    assert web_render._checkpoint_id(*args, model_options=alternative) != existing


def test_trim_preview_uses_selected_frames_without_clone_processing(tmp_path):
    clip = tmp_path / "reference.wav"
    sr = 16000
    original = np.repeat([0, 0.25, 0.5, 0], sr // 4)
    soundfile.write(clip, original, sr)

    output = Path(web_models.preview_model_reference(str(clip), trim_start=0.25, trim_end=0.75))
    try:
        data, sample_rate = soundfile.read(output)
        assert sample_rate == sr
        np.testing.assert_array_equal(data, original[sr // 4:3 * sr // 4])
    finally:
        output.unlink()

    with pytest.raises(web_models.ModelRunError, match="within"):
        web_models.preview_model_reference(str(clip), trim_start=1.1)


def test_voice_settings_remain_hashable_across_worker_json_and_change_checkpoints():
    voice = web_render.RenderVoice(
        model_instructions=(("z", "Last"), ("a", "First")), presets=(("pocket_local", "alba"),),
    )
    parsed = web_render.RenderVoice(**json.loads(json.dumps(asdict(voice))))
    assert parsed == voice
    assert hash(parsed) == hash(voice)
    assert parsed.model_instructions == (("a", "First"), ("z", "Last"))

    defaults = web_render.RenderVoice()
    variations = [
        defaults, replace(defaults, name="Storyteller"), replace(defaults, trim_start=0.1),
        replace(defaults, trim_end=2.0), replace(defaults, pan=0.5), replace(defaults, volume=75),
        replace(defaults, room_reverb=RoomReverb.SMALL_DRR22.id), replace(defaults, max_reference_silence=0.5),
        replace(defaults, presets=voice.presets), voice,
    ]
    ids = {web_render._checkpoint_id(Book([]), [item], "model", "en", False, 1) for item in variations}
    assert len(ids) == len(variations)


def test_book_checkpoint_retains_default_checksum_and_tracks_validation_and_substitution_order():
    voice = web_render.RenderVoice()
    book = Book([])
    baseline = {
        "version": 1, "book": book_to_project_text_json_dict(book), "voices": [["", "", ""]],
        "model": "model", "catalog": web_render.Tts._config_fingerprint, "language": "en",
        "validate_audio": True, "batch_size": 1,
    }
    old = hashlib.sha256(json.dumps(baseline, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    args = (book, [voice], "model", "en", True, 1)
    assert web_render._checkpoint_id(*args) == old
    assert web_render._checkpoint_id(*args, strictness=Strictness.get_recommended_default("en")) == old
    assert web_render._checkpoint_id(*args, runtime_settings=WebRuntimeSettings(save_generation_logs=True)) == old
    other_bitrate = next(value for value in AAC_BITRATES if value != AAC_BITRATE_DEFAULT)
    assert web_render._checkpoint_id(*args, runtime_settings=WebRuntimeSettings(aac_bitrate=other_bitrate)) == old
    assert web_render._checkpoint_id(*args, runtime_settings=WebRuntimeSettings(
        whisper_variant=SttVariant.get_default().id,
    )) == old
    variations = [
        old, web_render._checkpoint_id(*args, strictness=Strictness.HIGH),
        web_render._checkpoint_id(*args, max_retries=0),
        web_render._checkpoint_id(*args, word_substitutions={"alpha": "beta", "beta": "gamma"}),
        web_render._checkpoint_id(*args, word_substitutions={"beta": "gamma", "alpha": "beta"}),
        web_render._checkpoint_id(*args, runtime_settings=WebRuntimeSettings(force_cpu=True)),
        web_render._checkpoint_id(*args, runtime_settings=WebRuntimeSettings(
            whisper_variant=SttVariant.LARGE_V3.id, whisper_device=SttConfig.CPU_INT8FLOAT32.id,
        )),
    ]
    assert len(set(variations)) == len(variations)


@pytest.mark.parametrize("action", ["sample", "book"])
def test_web_model_options_round_trip_through_worker_json(monkeypatch, tmp_path, action):
    options = {"temperature": 0.65, "seed": 1024}
    book = Book([BookSection(phrase_groups=[PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])])])
    manual = web_render.RenderVoice(name="Line voice")
    session = BookSession(
        book, str(tmp_path), project_language="", word_substitutions={"rue": "roo"}, line_voices={7: manual},
    )
    controls = dict(first_line=1, last_line=1, regenerate=True, auto_concat=False, strictness="high", max_retries=3)
    runtime = WebRuntimeSettings(force_cpu=True, whisper_variant=SttVariant.LARGE_V3.id, save_generation_logs=True)
    sample = tmp_path / "recorded.wav"
    sample.touch()

    def render(*arguments, model_options, **callbacks):
        assert model_options == options
        assert callbacks["runtime_settings"] == runtime
        if action == "book":
            assert arguments[0].language_code == ""
            assert arguments[0].word_substitutions == {"rue": "roo"}
            assert arguments[0].line_voices == {7: manual}
            assert {key: callbacks[key] for key in controls} == controls
            return None
        return str(sample)

    def run(command, **_callbacks):
        directory = Path(command[-1])
        assert json.loads((directory / "input.json").read_text(encoding="utf-8"))["action"] == action
        web_models._worker(directory)
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(web_render, f"render_{action}", render)
    monkeypatch.setattr(web_render.ModelWorker, "shutdown", lambda: None)
    monkeypatch.setattr(web_models.signal, "signal", lambda *_: None)
    monkeypatch.setattr(web_models, "_python_for", lambda _model: Path(sys.executable))
    monkeypatch.setattr(web_models.subprocess, "Popen", run)
    if action == "sample":
        output = web_models.render_model_sample(
            "venv-breeze", "Hello.", web_render.RenderVoice(), model_options=options, runtime_settings=runtime,
        )
        assert output == str(sample)
    else:
        assert web_models.render_model_book(
            "venv-breeze", session, {}, model_options=options, runtime_settings=runtime, **controls,
        ) is None


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


def test_cast_uses_shared_voices_and_falls_back_to_the_named_narrator(monkeypatch):
    lines = ("Opening", "A named speaker", "A manual voice", "Unknown", "Uncast speaker",
             "An unidentified woman", "An unidentified man", "Another unidentified man", "The custodian")
    book = Book([BookSection(phrase_groups=[
        PhraseGroup(phrases=[Phrase(text=text, reason=Reason.SENTENCE)]) for text in lines
    ])])
    speakers = {
        1: "ALEX",
        2: "ALEX",
        3: "UNKNOWN",
        4: "SAM",
        5: "other:  Unidentified   Female",
        6: "OTHER:unidentified male",
        7: "OTHER:UNIDENTIFIED_MALE",
        8: "OTHER:Custodian",
    }
    monkeypatch.setattr(AttributionsFile, "speaker_of_each_line", lambda _self, _book: speakers)

    manual = web_render.RenderVoice(name="Line voice")
    book.phrase_groups[2].voice_index = 12
    roster = Roster((
        RosterCharacter("NINA", "Nina", is_narrator=True),
        RosterCharacter("ALEX", "Alex"),
        RosterCharacter("SAM", "Sam"),
    ))
    session = BookSession(book, "", roster=roster, line_voices={12: manual})
    nina = web_render.RenderVoice(instructions="Calm")
    alex = web_render.RenderVoice(instructions="Quiet")
    female = web_render.RenderVoice(instructions="Measured")
    male = web_render.RenderVoice(instructions="Warm")
    cast = {
        "NINA": nina, "ALEX": alex, "NARRATOR": web_render.RenderVoice(),
        "OTHER:unidentified female": female, "OTHER:unidentified male": male,
    }

    cast_book, voices = web_render._cast_book(session, cast)
    assert voices == [nina, alex, manual, female, male]
    assert [group.voice_index for group in cast_book.phrase_groups] == [0, 1, 2, 0, 0, 3, 4, 4, 0]
    assert [group.voice_index for group in book.phrase_groups] == [0, 0, 12, 0, 0, 0, 0, 0, 0]
    labels = web_render._cast_voice_labels(session, cast, voices)
    assert "unidentified female" in labels[3].casefold()
    assert "unidentified male" in labels[4].casefold()
    assert "other:" not in labels[3].casefold()
    assert "other:" not in labels[4].casefold()

    cast_book, voices = web_render._cast_book(session, {"NINA": nina, "ALEX": web_render.RenderVoice()})
    assert voices == [nina, manual]
    assert [group.voice_index for group in cast_book.phrase_groups] == [0, 0, 1, 0, 0, 0, 0, 0, 0]


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
    session = BookSession(book, str(tmp_path))
    submissions = []
    temperatures = []
    pronunciations = []
    statuses = iter([
        GenerationTerminalStatus.CANCELLED,
        GenerationTerminalStatus.COMPLETED,
        GenerationTerminalStatus.CANCELLED,
        GenerationTerminalStatus.CANCELLED,
    ])
    time = np.arange(24000, dtype=np.float32) / 24000
    sample = Sound(0.12 * np.sin(2 * np.pi * time * 440), 24000)

    def submit(*, state, indices, batch_size, is_regen, staging_dir, **_):
        assert not is_regen and not staging_dir
        submissions.append(indices)
        temperatures.append(state.project.get_model_setting(model.id, "temperature"))
        pronunciations.append(state.project.word_substitutions)
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

    with pytest.raises(web_render.WebRenderError, match="stopped"):
        updated = replace(session, word_substitutions={"fox": "focks"})
        web_render.render_book(updated, {}, validate_audio=False, model_options={"temperature": 0.65})
    assert submissions == [{0, 1}, {1}, {0, 1}, {0, 1}]
    assert temperatures[-1] == 0.65
    assert pronunciations == [{}, {}, {}, {"fox": "focks"}]


def test_selective_generation_shares_checkpoint_and_regeneration_retains_existing_takes(monkeypatch, tmp_path):
    from tts_audiobook_tool.app_types.validation_result import SkippedResult
    from tts_audiobook_tool.generate_util import GenerateUtil
    from tts_audiobook_tool.project_support.sound_segment_util import get_segment_stt_info_path

    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render, "mark_interactive_main", lambda: None)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    book = Book([BookSection(phrase_groups=[
        PhraseGroup([Phrase(text, Reason.SENTENCE)]) for text in ("One.", "Two.", "Three.")
    ])])
    session = BookSession(book, str(tmp_path))
    requests = []
    events = []
    statuses = iter([
        GenerationTerminalStatus.COMPLETED,
        GenerationTerminalStatus.COMPLETED,
        GenerationTerminalStatus.CANCELLED,
        GenerationTerminalStatus.COMPLETED,
        GenerationTerminalStatus.COMPLETED,
    ])
    time = np.arange(24000, dtype=np.float32) / 24000
    sample = Sound(0.12 * np.sin(2 * np.pi * time * 400), 24000)
    export_path = None

    def submit(*, state, indices, batch_size, is_regen, staging_dir, **_):
        requests.append((indices, is_regen))
        assert state.project.strictness is Strictness.HIGH
        assert state.project.max_retries == 0
        assert bool(staging_dir) is is_regen
        if is_regen and export_path is not None:
            assert export_path.is_file()
        for index in indices:
            error, path = GenerateUtil.save_sound_and_timing_json(
                state, state.project.phrase_groups[index], index, SkippedResult(sample, "Validation skipped"),
                is_real_time=False, dir_path=staging_dir or None,
            )
            assert not error
            if staging_dir:
                get_segment_stt_info_path(path).write_text("{}", encoding="utf-8")
        return "job"

    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", submit)
    monkeypatch.setattr(
        web_render.ModelWorker, "get_event", lambda **_: GenerationFinished("job", next(statuses), ""),
    )
    options = dict(validate_audio=False, strictness="high", max_retries=0, on_progress=lambda *args: events.append(args))
    assert web_render.render_book(session, {}, first_line=2, last_line=2, **options) is None
    assert events == [(0, 1), (1, 1)]
    assert web_render.render_book(session, {}, first_line=2, last_line=2, auto_concat=False, **options) is None
    assert requests == [({1}, False)]

    export_path = Path(web_render.render_book(session, {}, **options))
    before = export_path.read_bytes()
    directory = export_path.parent
    originals = sorted((directory / "segments").glob("*.flac"))
    assert len(originals) == 3
    assert len(list((tmp_path / "renders").iterdir())) == 1

    with pytest.raises(web_render.WebRenderError, match="stopped"):
        web_render.render_book(session, {}, first_line=2, last_line=2, regenerate=True, **options)
    assert export_path.read_bytes() == before
    assert sorted((directory / "segments").glob("*.flac")) == originals

    replace_path = Path.replace

    def fail_audio_move(source, target):
        if source.suffix == ".flac" and "[take " in Path(target).name:
            raise OSError("test failed audio move")
        return replace_path(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_audio_move)
        with pytest.raises(web_render.WebRenderError, match="could not be saved"):
            web_render.render_book(session, {}, first_line=2, last_line=2, regenerate=True, **options)
    assert export_path.read_bytes() == before
    assert not list((directory / "segments").glob("* [take *"))

    assert web_render.render_book(
        session, {}, first_line=2, last_line=2, regenerate=True, auto_concat=False, **options,
    ) is None
    assert not export_path.exists()
    takes = set((directory / "segments").glob("*.flac"))
    assert set(originals) < takes
    new, = takes - set(originals)
    assert new.name.startswith("[00002]")
    assert soundfile.info(new).frames == sample.sr
    assert get_segment_stt_info_path(new).is_file()
    assert web_render.render_book(session, {}, **options) == str(export_path)
    assert export_path.is_file()
    assert requests == [({1}, False), ({0, 2}, False), ({1}, True), ({1}, True), ({1}, True)]


def test_book_uses_selected_validation_prefs_and_saves_native_generation_log(monkeypatch, tmp_path):
    from faster_whisper import utils
    from tts_audiobook_tool.app_types import DeviceType
    from tts_audiobook_tool.app_types.validation_result import SkippedResult
    from tts_audiobook_tool.generate_util import GenerateUtil
    from tts_audiobook_tool.prefs import Prefs
    from tts_audiobook_tool.stt import Stt

    model = TtsModelType.require_by_id("chatterbox_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    monkeypatch.setattr(web_render.Tts, "_force_cpu", False)
    monkeypatch.setattr(Prefs, "save", lambda *_: pytest.fail("Console preference write"))
    monkeypatch.setattr(Stt, "should_use_mlx_whisper", lambda: False)
    cache = tmp_path / "whisper-cache"
    cache.mkdir()
    for filename in ("model.bin", "config.json", "tokenizer.json"):
        (cache / filename).touch()
    native_lookups = []

    def lookup(variant, *, local_files_only):
        native_lookups.append((variant, local_files_only))
        return str(cache)

    monkeypatch.setattr(utils, "download_model", lookup)
    book = Book([BookSection(phrase_groups=[PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])])])
    session = BookSession(book, str(tmp_path))
    runtime = WebRuntimeSettings(
        whisper_variant=SttVariant.LARGE_V3.id, whisper_device=SttConfig.CPU_INT8FLOAT32.id,
        force_cpu=True, save_generation_logs=True,
    )
    events = iter([
        ConsoleOutput("another-job", "stdout", "Do not record this job.\n"),
        GenerationUpdate("job", GenerationPhase("Preparing models")),
        GenerationUpdate("job", GenerationPhase("Preparing models")),
        ConsoleOutput("job", "stdout", "\x1b[32mGenerated audio in 1.0s\x1b[0m\r\n"),
        GenerationFinished("job", GenerationTerminalStatus.COMPLETED, ""),
    ])
    console = []

    def submit(*, state, indices, suppress_validation_passes, **_):
        assert suppress_validation_passes is True
        assert indices == {0}
        assert state.prefs.stt_variant is SttVariant.LARGE_V3
        assert state.prefs.stt_config is SttConfig.CPU_INT8FLOAT32
        assert state.prefs.save_gen_log
        assert state.prefs.tts_force_cpu
        assert web_render.Tts.get_best_supported_device_type(model) is DeviceType.CPU
        time = np.arange(24000, dtype=np.float32) / 24000
        sample = Sound(0.12 * np.sin(2 * np.pi * time * 400), 24000)
        error, _ = GenerateUtil.save_sound_and_timing_json(
            state, state.project.phrase_groups[0], 0, SkippedResult(sample, "Validation mocked"), is_real_time=False,
        )
        assert not error
        return "job"

    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: next(events))
    monkeypatch.setattr(web_render.ModelWorker, "drain_events", lambda: [
        ConsoleOutput("another-job", "stdout", "Do not record this either.\n"),
        ConsoleOutput("job", "stdout", "Lines saved: 1 (all)\n"),
    ])
    assert web_render.render_book(session, {}, runtime_settings=runtime, auto_concat=False, on_console=console.append) is None
    assert native_lookups == [(SttVariant.LARGE_V3.id, True)]
    assert console.count("Preparing models\n") == 1
    assert console[-3:] == ["Lines saved: 1 (all)\n", "Generation completed.\n", "Automatic export is off.\n"]
    assert "Do not record" not in "".join(console)
    log = latest_generation_log(str(tmp_path))
    assert log is not None
    assert Path(log).read_text(encoding="utf-8") == "Generated audio in 1.0s\nLines saved: 1 (all)\nGeneration completed.\n"


def test_automatic_export_rebuilds_for_bitrate_without_regenerating_takes(monkeypatch, tmp_path):
    from tts_audiobook_tool.app_types.validation_result import SkippedResult
    from tts_audiobook_tool.concat_util import ConcatUtil
    from tts_audiobook_tool.generate_util import GenerateUtil

    model = TtsModelType.require_by_id("breeze_tts_2_local")
    monkeypatch.setattr(web_render, "_detect_model", lambda: model)
    monkeypatch.setattr(web_render.Tts, "_available_local_models", (model,))
    book = Book([BookSection(phrase_groups=[PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])])])
    session = BookSession(book, str(tmp_path))
    generate = []
    bitrates = []
    native_export = ConcatUtil.make_file

    def submit(*, state, indices, **_):
        generate.append(indices)
        time = np.arange(24000, dtype=np.float32) / 24000
        sound = Sound(0.12 * np.sin(2 * np.pi * time * 400), 24000)
        error, _ = GenerateUtil.save_sound_and_timing_json(
            state, state.project.phrase_groups[0], 0, SkippedResult(sound, "Synthesis mocked"), is_real_time=False,
        )
        assert not error
        return "job"

    def make_file(*, state, **kwargs):
        bitrates.append(state.prefs.aac_bitrate)
        return native_export(state=state, **kwargs)

    monkeypatch.setattr(web_render.ModelWorker, "submit_generation", submit)
    monkeypatch.setattr(web_render.ModelWorker, "get_event", lambda **_: GenerationFinished(
        "job", GenerationTerminalStatus.COMPLETED, "",
    ))
    monkeypatch.setattr(ConcatUtil, "make_file", make_file)
    output = Path(web_render.render_book(session, {}, validate_audio=False))
    previous = output.read_bytes()
    files = sorted((output.parent / "segments").glob("*.flac"))
    assert len(files) == 1
    rendered_audio = files[0].read_bytes()

    # An old default export had no marker and should still be reused.
    output.with_suffix(".bitrate.json").unlink()
    assert web_render.render_book(session, {}, validate_audio=False) == str(output)
    assert bitrates == [AAC_BITRATE_DEFAULT]

    other_bitrate = next(value for value in AAC_BITRATES if value != AAC_BITRATE_DEFAULT)
    runtime = WebRuntimeSettings(aac_bitrate=other_bitrate)
    replace = Path.replace

    def fail_audio_publication(source, target):
        if Path(target) == output:
            raise OSError("test audio publication interrupted")
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_audio_publication)
        with pytest.raises(web_render.WebRenderError, match="audio could not be exported"):
            web_render.render_book(session, {}, validate_audio=False, runtime_settings=runtime)
    assert output.read_bytes() == previous
    assert not web_render._automatic_export_matches(output, other_bitrate)

    assert web_render.render_book(session, {}, validate_audio=False, runtime_settings=runtime) == str(output)
    assert web_render.render_book(session, {}, validate_audio=False, runtime_settings=runtime) == str(output)
    assert bitrates == [AAC_BITRATE_DEFAULT, other_bitrate, other_bitrate]
    assert generate == [{0}]
    assert sorted((output.parent / "segments").glob("*.flac")) == files
    assert files[0].read_bytes() == rendered_audio
    assert len(list((tmp_path / "renders").iterdir())) == 1


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
            assert len(data) == 8000
            assert language == "en"
            assert word_timestamps
            return [ConcreteSegment(0.0, 0.3, "A clear word.", [
                ConcreteWord(0.0, 0.2, " Clear", 0.9),
                ConcreteWord(0.2, 0.3, " faint", 0.1),
            ])], None

    monkeypatch.setattr(utils, "download_model", lookup)
    monkeypatch.setattr(stt, "FasterWhisperAdapter", FakeWhisper)
    monkeypatch.setattr(web_render, "mark_interactive_main", lambda: None)
    assert web_render.transcribe_reference(
        str(clip), language_code="en-US", trim_start=0.25, trim_end=0.75,
    ) == "Clear"
