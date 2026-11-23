import asyncio
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event
import wave

import pytest

gr = pytest.importorskip("gradio", minversion="6.17.3")
from gradio.state_holder import SessionState

from tts_audiobook_tool import gradio_app, web_projects, web_voice_panel, web_voices
from tts_audiobook_tool.app_types import Book, BookSection, RoomReverb
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generation_events import BookRenderTelemetry
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.web_backend import BookSession
from tts_audiobook_tool.web_models import ModelRunError
from tts_audiobook_tool.web_render import RenderVoice


class VoiceApp:
    """Send configured browser events through Gradio's real preprocessing and output handling."""

    def __init__(self):
        self.app = gradio_app.build_app()
        self.app.has_launched = True
        self.state = SessionState(self.app)
        self.values = {component["id"]: component["props"].get("value") for component in self.app.config["components"]}

    def event(self, name, target=None):
        return next(
            fn for fn in self.app.fns.values()
            if fn.name == name
            and (
                target is None
                or any(getattr(self.app.blocks[block], "label", None) == target for block, _ in fn.targets)
            )
        )

    async def run(self, fn, *values):
        inputs = list(values) if values else [self.values[component._id] for component in fn.inputs]
        request = gr.Request(session_hash="test-voice-workflow")
        result = await self.app.process_api(fn, inputs, self.state, request=request)
        frames = []
        while True:
            if result["is_generating"] or result["iterator"] is not None or not frames:
                frames.append(result["data"])
                for component, value in zip(fn.outputs, result["data"], strict=True):
                    if component.stateful:
                        continue
                    if isinstance(value, dict) and value.get("__type__") == "update":
                        if "value" not in value:
                            continue
                        value = value["value"]
                    self.values[component._id] = value
            if not result["is_generating"]:
                return frames
            result = await self.app.process_api(fn, inputs, self.state, request=request, iterator=result["iterator"])


def _uploaded_audio(directory):
    directory.mkdir()
    path = directory / "reference.wav"
    with wave.open(str(path), "wb") as clip:
        clip.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        clip.writeframes(b"\x00\x00" * 8_000)
    return {"path": str(path), "meta": {"_type": "gradio.FileData"}}


def test_live_console_and_overall_audio_speed_update_before_render_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(gradio_app, "installed_model_choices", lambda: [("Breeze TTS 2", "venv-breeze")])
    halfway = Event()
    finish = Event()

    def render(_model, _session, _voices, *, on_console, on_telemetry, **_kwargs):
        on_console("Saved audio for the first batch.\n")
        on_telemetry(BookRenderTelemetry(processed=2, total=4, speed_multiplier=10.0))
        halfway.set()
        assert finish.wait(10)
        on_telemetry(BookRenderTelemetry(processed=4, total=4, speed_multiplier=8.0))

    monkeypatch.setattr(gradio_app, "render_model_book", render)
    ui = VoiceApp()
    generate = ui.event("generate_audiobook")
    refresh = ui.event("refresh_activity")
    assert refresh.queue is False
    workspace_row = next(
        node for node in ui.app.config["layout"]["children"]
        if any(
            getattr(ui.app.blocks[child["id"]], "elem_id", None) == "voice-settings"
            for child in node.get("children", [])
        )
    )
    _, workspace = workspace_row["children"]
    tabs, console_form, *_ = workspace["children"]
    console = refresh.outputs[0]
    assert isinstance(ui.app.blocks[tabs["id"]], gr.Tabs)
    assert console._id in {node["id"] for node in console_form["children"]}
    assert console.visible and console.label and console.placeholder
    book = Book([BookSection([PhraseGroup([Phrase(f"Line {i}.", Reason.SENTENCE)]) for i in range(4)])])
    ui.state[generate.inputs[0]._id] = BookSession(book, str(tmp_path))

    async def exercise():
        job = asyncio.create_task(ui.run(generate))
        try:
            assert await asyncio.to_thread(halfway.wait, 10)
            (midway,) = await ui.run(refresh)
            assert "Saved audio for the first batch." in midway[0]
            assert "2 of 4 segments · 50%" in midway[1]["value"]
            assert "10.0x" in midway[1]["value"]
            assert '<progress aria-label="Overall audiobook progress" max="4" value="2">' in midway[1]["value"]

            viewer = gr.Request(session_hash="another-session")
            other = await ui.app.process_api(refresh, [None], SessionState(ui.app), request=viewer)
            assert all("value" not in item for item in other["data"][:2])
        finally:
            finish.set()
            await job
        (completed,) = await ui.run(refresh)
        assert "Selected lines are ready" in completed[0]
        assert "4 of 4 segments · 100%" in completed[1]["value"]
        assert "8.0x" in completed[1]["value"]

    asyncio.run(exercise())


def test_reference_upload_and_recording_transcribe_with_editable_fallback(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    reference = _uploaded_audio(upload_dir)
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(upload_dir))
    monkeypatch.setattr(gradio_app, "installed_model_choices", lambda: [])
    calls = []
    trim_calls = []

    def transcribe(path, *, language_code, trim_start, trim_end):
        calls.append((Path(path).read_bytes(), language_code, trim_start, trim_end))
        if language_code == "de":
            raise ModelRunError("Recognizer unavailable.")
        return "Recognized sample."

    monkeypatch.setattr(web_voice_panel, "transcribe_model_reference", transcribe)

    def preview(path, *, trim_start, trim_end):
        trim_calls.append((trim_start, trim_end))
        result = tmp_path / "trimmed.wav"
        result.write_bytes(Path(path).read_bytes())
        return str(result)

    monkeypatch.setattr(web_voice_panel, "preview_model_reference", preview)
    ui = VoiceApp()

    async def exercise():
        narrator = ui.event("transcribe_new", "Reference audio")
        character = ui.event("transcribe_new", "Character reference audio")
        assert {kind for _, kind in narrator.targets} == {"upload", "stop_recording"}
        assert {kind for _, kind in character.targets} == {"upload", "stop_recording"}

        start, done = await ui.run(narrator, reference, "en")
        assert start[0] == gr.update(value="", interactive=False)
        assert done[0] == gr.update(value="Recognized sample.", interactive=True)
        assert done[1]["visible"] is False

        start, done = await ui.run(character, reference, "de")
        assert start[0] == gr.update(value="", interactive=False)
        assert done[0] == gr.update(interactive=True)
        assert done[1]["visible"] is True

        retry = ui.event("result")
        (done,) = await ui.run(retry, reference, "en", 0.1, 0.5)
        assert done[0]["value"] == "Recognized sample."
        ui.values[retry.outputs[0]._id] = "Manually edited transcript."
        await ui.run(retry, reference, "de", 0.1, 0.5)
        assert ui.values[retry.outputs[0]._id] == "Manually edited transcript."

        (done,) = await ui.run(ui.event("preview_trim"), reference, 0.1, 0.5)
        assert done[0]["visible"] is True
        assert Path(done[0]["value"]["path"]).is_file()
        assert not (tmp_path / "trimmed.wav").exists()

    asyncio.run(exercise())
    assert calls == [
        (Path(reference["path"]).read_bytes(), "en", 0, None),
        (Path(reference["path"]).read_bytes(), "de", 0, None),
        (Path(reference["path"]).read_bytes(), "en", 0.1, 0.5),
        (Path(reference["path"]).read_bytes(), "de", 0.1, 0.5),
    ]
    assert trim_calls == [(0.1, 0.5)]


def test_library_selection_reaches_samples_cast_and_saved_project(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    reference = _uploaded_audio(upload_dir)
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(upload_dir))
    monkeypatch.setattr(gradio_app, "installed_model_choices", lambda: [("Breeze TTS 2", "venv-breeze")])
    voice_root = tmp_path / "voices"
    project_root = tmp_path / "projects"
    monkeypatch.setattr(web_voice_panel, "list_voices", partial(web_voices.list_voices, root=voice_root))
    monkeypatch.setattr(web_voice_panel, "load_voice", partial(web_voices.load_voice, root=voice_root))
    monkeypatch.setattr(gradio_app, "load_voice", partial(web_voices.load_voice, root=voice_root))
    monkeypatch.setattr(gradio_app, "matching_voice_id", partial(web_voices.matching_voice_id, root=voice_root))
    for method in ("create_project", "list_projects", "load_project"):
        monkeypatch.setattr(gradio_app, method, partial(getattr(web_projects, method), root=project_root))

    narrator = RenderVoice(
        reference["path"], "Edited reference.", name="Library narrator", trim_start=0.1, trim_end=0.8,
        pan=-0.45, volume=85, room_reverb=RoomReverb.SMALL_DRR22.id,
        model_instructions=(("breeze_tts_2_local", "Read gently."), ("future_model", "Keep this instruction.")),
        presets=(("qwen3tts_local", "vivian"), ("pocket_local", "alba"), ("future_model", "future-speech")),
        max_reference_silence=0.4,
    )
    actor = replace(narrator, name="Rue voice", pan=0.4)
    saved_narrator = web_voices.create_voice(narrator, root=voice_root, upload_root=upload_dir)
    saved_actor = web_voices.create_voice(actor, root=voice_root, upload_root=upload_dir)
    book = Book([BookSection([
        PhraseGroup([Phrase("Hello, Aya.", Reason.PARAGRAPH)], is_dialog=True),
        PhraseGroup([Phrase("Time to go.", Reason.PARAGRAPH)], is_dialog=True),
        PhraseGroup([Phrase("I'll be right there.", Reason.SECTION_BREAK)], is_dialog=True),
    ])])
    session = BookSession(book, str(tmp_path))
    roster = Roster((RosterCharacter("AYA", "Aya", is_narrator=True), RosterCharacter("RUE", "Rue"), RosterCharacter("MINA", "Mina")))
    unidentified_female = "OTHER:unidentified female"
    unidentified_male = "OTHER:unidentified male"
    epub = upload_dir / "book.epub"
    epub.write_bytes(b"test EPUB")
    monkeypatch.setattr(gradio_app.TextImportPanel, "import_book", staticmethod(lambda *_: session))
    monkeypatch.setattr(gradio_app, "identify_characters", lambda session, *_args, **_kwargs: replace(session, roster=roster))
    rendered = []
    attributions = []
    rendered_books = []

    def assign_dialogue(session, *_args, on_progress, **_kwargs):
        attributions.append(session)
        on_progress(1, 1)
        assigned_range = RangeAttribution(0, 3, UtteranceUtil.range_text_hash(book, 0, 3), (
            UtteranceAttribution((0,), "RUE"), UtteranceAttribution((1,), "MINA"),
            UtteranceAttribution((2,), unidentified_female),
        ))
        return replace(session, attributions=AttributionsFile((assigned_range,)))

    def render_sample(model, text, voice, **kwargs):
        rendered.append(voice)
        return reference["path"]

    def render_book(model, session, voices, **kwargs):
        rendered_books.append(voices)

    monkeypatch.setattr(gradio_app, "attribute_dialog", assign_dialogue)
    monkeypatch.setattr(gradio_app, "render_model_sample", render_sample)
    monkeypatch.setattr(gradio_app, "render_model_book", render_book)
    monkeypatch.setattr(
        web_voice_panel, "transcribe_model_reference", lambda *_args, **_kw: pytest.fail("New media is required."),
    )
    ui = VoiceApp()

    async def exercise():
        await ui.run(ui.event("refresh_library"))
        select = ui.event("select_voice")
        await ui.run(select, saved_actor.voice_id, None)
        compare = ui.event("compare_samples")
        await ui.run(compare, ["venv-breeze"], "A preview.", "en", *[ui.values[c._id] for c in compare.inputs[3:]])
        assert rendered == [replace(actor, reference_path=rendered[0].reference_path)]

        load = ui.event("load_book")
        fields = [ui.values[component._id] for component in load.inputs]
        fields[3] = {"path": str(epub), "meta": {"_type": "gradio.FileData"}}
        (loaded,) = await ui.run(load, *fields)
        assert loaded[4]["choices"] == [["Narrator", "NARRATOR"]]
        identify = ui.event("find_characters")
        arguments = [ui.values[component._id] for component in identify.inputs]
        arguments[1] = "fixture-key"
        _, identified = await ui.run(identify, *arguments)
        assert identified[2]["value"] == {"headers": ["Character"], "data": [["Aya"], ["Rue"], ["Mina"]], "metadata": None}
        assert identified[3]["choices"] == [
            ["Narrator", "AYA"], ["Rue", "RUE"], ["Mina", "MINA"],
            ["Unidentified female", unidentified_female], ["Unidentified male", unidentified_male],
        ]
        assert identified[4]["value"].startswith("Identified 3 characters.")

        show_cast = ui.event("show_cast_voice")
        save_assignment = ui.event("save_cast_assignment")
        generate = ui.event("generate_audiobook")
        cast_state = save_assignment.inputs[1]._id
        cast_role = show_cast.inputs[0]._id
        voice_picker = next(component for component in save_assignment.inputs if getattr(component, "label", None) == "Choose a saved voice")
        cast_status = next(component for component in save_assignment.outputs if getattr(component, "label", None) == "Casting status")
        assert all((voice_picker._id, "input") not in fn.targets for fn in ui.app.fns.values())

        async def pick_role(role):
            ui.values[cast_role] = role
            await ui.run(show_cast)

        await pick_role("RUE")
        assert ui.values[cast_status._id] == "Rue uses the narrator voice by default."
        ui.values[voice_picker._id] = saved_actor.voice_id
        assert ui.state[cast_state] == {}
        await ui.run(save_assignment)
        assert ui.state[cast_state]["RUE"].name == actor.name
        assert ui.values[cast_status._id] == "Saved Rue voice for Rue."

        await pick_role("MINA")
        assert ui.values[cast_status._id] == "Mina uses the narrator voice by default."
        assert "MINA" not in ui.state[cast_state]
        await pick_role(unidentified_male)
        assert ui.values[cast_status._id] == "Unidentified male uses the narrator voice by default."
        await pick_role(unidentified_female)
        assert ui.values[cast_status._id] == "Unidentified female uses the narrator voice by default."
        ui.values[voice_picker._id] = saved_actor.voice_id
        await ui.run(save_assignment)
        assert ui.values[cast_status._id] == "Saved Rue voice for Unidentified female."
        assert ui.state[cast_state][unidentified_female].name == actor.name
        await pick_role("RUE")
        assert ui.values[cast_status._id] == "Rue uses Rue voice."
        assert ui.values[voice_picker._id] == saved_actor.voice_id

        await pick_role("AYA")
        ui.values[voice_picker._id] = saved_narrator.voice_id
        assert ui.values[select.inputs[0]._id] == saved_actor.voice_id
        await ui.run(save_assignment)
        assert ui.values[cast_status._id] == "Saved Library narrator for the narrator."
        assert ui.values[select.inputs[0]._id] == saved_narrator.voice_id
        assert not attributions
        (blocked,) = await ui.run(generate)
        assert "Run Assign dialogue" in blocked[0]["value"]
        assert not rendered_books

        saved_cast = dict(ui.state[cast_state])
        assign_dialogue_event = ui.event("find_speakers")
        arguments = [ui.values[component._id] for component in assign_dialogue_event.inputs]
        arguments[1] = "fixture-key"
        await ui.run(assign_dialogue_event, *arguments)
        assert ui.state[cast_state] == saved_cast
        assert ui.values[cast_role] == "AYA"
        assert ui.values[voice_picker._id] == saved_narrator.voice_id
        await ui.run(generate)
        assert len(attributions) == len(rendered_books) == 1
        assert rendered_books[0] == {
            "NARRATOR": replace(narrator, reference_path=rendered_books[0]["NARRATOR"].reference_path),
            "RUE": saved_cast["RUE"], unidentified_female: saved_cast[unidentified_female],
        }
        await ui.run(compare, ["venv-breeze"], "A narrator preview.", "en", *[ui.values[c._id] for c in compare.inputs[3:]])
        assert rendered[-1] == replace(narrator, reference_path=rendered[-1].reference_path)

        save = ui.event("save_named_project")
        values = [ui.values[component._id] for component in save.inputs]
        values[0] = "Voice workflow"
        (result,) = await ui.run(save, *values)
        assert result[-1]["value"] == "Saved Voice workflow."
        project = web_projects.list_projects(root=project_root)[0]
        open_project = ui.event("open_project_snapshot")
        (reopened,) = await ui.run(open_project, project.project_id, None)
        assert len(reopened) == len(open_project.outputs)
        project_actor = ui.state[cast_state]["RUE"]
        assert project_actor.reference_path != saved_cast["RUE"].reference_path
        for role in ("RUE", unidentified_female):
            await pick_role(role)
            assert ui.values[voice_picker._id] == saved_actor.voice_id

        web_voices.save_voice(
            saved_actor.voice_id, replace(actor, transcript="Edited in the library only."),
            root=voice_root, upload_root=upload_dir,
        )
        await pick_role("RUE")
        assert ui.values[voice_picker._id] is None
        assert ui.values[cast_status._id] == "Rue uses a saved copy of Rue voice."
        assert ui.state[cast_state]["RUE"] == project_actor

        web_voices.delete_voice(saved_actor.voice_id, root=voice_root)
        await ui.run(open_project, project.project_id, None)
        await pick_role("RUE")
        assert ui.values[voice_picker._id] is None
        assert ui.values[cast_status._id] == "Rue uses a saved copy of Rue voice."
        assert ui.state[cast_state]["RUE"] == project_actor
        await ui.run(compare, ["venv-breeze"], "A second preview.", "en", *[ui.values[c._id] for c in compare.inputs[3:]])
        assert len(rendered) == 3
        assert rendered[-1] == replace(narrator, reference_path=rendered[-1].reference_path)

        opened = web_projects.load_project(project.project_id, root=project_root)
        try:
            assert opened.narrator == replace(narrator, reference_path=opened.narrator.reference_path)
            assert opened.voices["RUE"] == replace(actor, reference_path=opened.voices["RUE"].reference_path)
            assert opened.voices[unidentified_female] == replace(actor, reference_path=opened.voices[unidentified_female].reference_path)
            assert "MINA" not in opened.voices
            assert Path(opened.voices["RUE"].reference_path).is_file()
        finally:
            gradio_app.close_book_session(opened.session)

    asyncio.run(exercise())


def test_model_install_settings_and_batch_reach_render_callbacks(tmp_path, monkeypatch):
    if not gradio_app.model_profiles()[0].installable:
        pytest.skip("Local setup is supported on Linux")
    reference = _uploaded_audio(tmp_path / "uploads")
    installed = [("Breeze TTS 2", "venv-breeze")]
    monkeypatch.setattr(gradio_app, "installed_model_choices", lambda: installed[:])
    monkeypatch.setattr("tts_audiobook_tool.web_model_panel.recommended_batch_size",
                        lambda name, **_: 5 if name == "venv-qwen3tts" else 10)
    setup = []
    setup_progress = []
    rendered = []

    def install(model, *, on_progress, cancel_event):
        assert not cancel_event.is_set()
        on_progress(gradio_app.ModelSetupProgress(3, 5, "Verify the environment"))
        setup.append(model)
        installed.append(("Qwen3-TTS", model))
        on_progress(gradio_app.ModelSetupProgress(5, 5, "Setup complete"))

    def render_book(model, session, voices, **kwargs):
        rendered.append((model, kwargs["model_options"], kwargs["batch_size"]))
        return reference["path"]

    def render_sample(model, passage, voice, **kwargs):
        rendered.append((model, kwargs["model_options"]))
        return reference["path"]

    monkeypatch.setattr(gradio_app, "install_model", install)
    monkeypatch.setattr(gr.Progress, "__call__", lambda _self, value, *, desc: setup_progress.append((value, desc)))
    monkeypatch.setattr(gradio_app, "render_model_book", render_book)
    monkeypatch.setattr(gradio_app, "render_model_sample", render_sample)
    ui = VoiceApp()

    async def exercise():
        install_event = ui.event("install_selected_model")
        assert install_event.show_progress_on == [install_event.outputs[8]]
        assert install_event.show_progress_on[0].label == "Installation status"
        sample = Book([BookSection([PhraseGroup([Phrase("Hello.", Reason.SENTENCE)])])])
        ui.state[install_event.inputs[2]._id] = BookSession(sample, str(tmp_path))
        _, changed = await ui.run(install_event, "venv-qwen3tts", ["venv-breeze"], None, True, None, False)
        assert changed[2]["value"] == "venv-qwen3tts"
        assert changed[3]["value"] == ["venv-breeze", "venv-qwen3tts"]
        assert changed[4] == 5
        assert changed[5]["interactive"]

        render_options = next(
            fn for fn in ui.app.fns.values()
            if fn.renderable is not None and getattr(fn.inputs[0], "label", None) == "Local speech models"
        )
        await ui.run(render_options, "venv-qwen3tts", None)
        apply = next(fn for fn in ui.state.blocks_config.fns.values() if fn.name == "apply_model_settings")
        settings = [component.value for component in apply.inputs[1:]]
        labels = [component.label for component in apply.inputs[1:]]
        settings[labels.index("Temperature")] = 0.7
        await ui.run(apply, None, *settings)
        assert ui.state[apply.inputs[0]._id] == {"venv-qwen3tts": {"temperature": 0.7}}
        await ui.run(ui.event("generate_audiobook"))
        compare = ui.event("compare_samples")
        await ui.run(compare, ["venv-breeze", "venv-qwen3tts"], "A preview.", "en",
                     *[ui.values[c._id] for c in compare.inputs[3:]])

        reset = next(fn for fn in ui.state.blocks_config.fns.values() if fn.name == "reset_model_settings")
        await ui.run(reset, None)
        assert ui.state[reset.inputs[0]._id] == {}

    asyncio.run(exercise())
    assert setup == ["venv-qwen3tts"]
    assert setup_progress[:2] == [
        ((3, 5), "Verify the environment (3/5 steps complete)"),
        ((5, 5), "Setup complete (5/5 steps complete)"),
    ]
    assert rendered == [
        ("venv-qwen3tts", {"temperature": 0.7}, 5),
        ("venv-breeze", None),
        ("venv-qwen3tts", {"temperature": 0.7}),
    ]


def test_first_text_import_keeps_the_abr_project_selected(tmp_path, monkeypatch):
    from tts_audiobook_tool.web_abr_project import AbrProject

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    audio = uploads / "template.m4b"
    audio.write_bytes(b"ABR fixture")
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(uploads))
    monkeypatch.setattr(gradio_app, "installed_model_choices", lambda: [("Breeze TTS 2", "venv-breeze")])
    for method in ("create_project", "list_projects", "load_project", "save_project"):
        monkeypatch.setattr(gradio_app, method, partial(getattr(web_projects, method), root=tmp_path / "projects"))
    empty = BookSession(Book([], title="Template"), str(tmp_path), project_language="fr",
                        word_substitutions={"rue": "roo"})
    template = AbrProject(empty, RenderVoice(name="Narrator"), "venv-breeze", {}, 3, "high", 4, False)
    monkeypatch.setattr(gradio_app, "read_abr_project", lambda *_: template)
    ui = VoiceApp()

    async def exercise():
        file_data = {"path": str(audio), "meta": {"_type": "gradio.FileData"}}
        await ui.run(ui.event("create_abr_project"), file_data, "New ABR project", None)
        load = ui.event("load_book")
        prior_id = ui.state[load.inputs[1]._id]
        values = [ui.values[component._id] for component in load.inputs]
        values[2] = "manual"
        values[5] = "Rue opened the door."
        await ui.run(load, *values)
        assert ui.state[load.inputs[1]._id] == prior_id
        assert ui.state[load.inputs[0]._id].language_code == "fr"
        assert ui.state[load.inputs[0]._id].word_substitutions == {"rue": "roo"}
        await ui.run(ui.event("save_loaded_project"))
        project = web_projects.load_project(prior_id, root=tmp_path / "projects")
        try:
            assert project.session.book.phrase_groups[0].text.strip() == "Rue opened the door."
            assert (project.batch_size, project.strictness, project.max_retries, project.auto_concat) == (3, "high", 4, False)
            assert project.narrator.name == "Narrator"
        finally:
            gradio_app.close_book_session(project.session)

    asyncio.run(exercise())


def test_compare_resolves_remote_selection_for_the_current_endpoint(tmp_path, monkeypatch):
    from tts_audiobook_tool.web_remote_models import REMOTE_MODEL_ID

    reference = _uploaded_audio(tmp_path / "uploads")
    monkeypatch.setattr(gradio_app, "installed_model_choices", lambda: [])
    selected = object()
    sent = []

    def resolve(endpoint, _state):
        assert endpoint == "http://127.0.0.1:8000"
        return selected

    def render_sample(model, _text, _voice, **kwargs):
        sent.append((model, kwargs["remote_selection"]))
        return reference["path"]

    monkeypatch.setattr(gradio_app.RemoteModelPanel, "selection_for", staticmethod(resolve))
    monkeypatch.setattr(gradio_app, "render_model_sample", render_sample)
    ui = VoiceApp()
    compare = ui.event("compare_samples")
    values = [ui.values[component._id] for component in compare.inputs]
    values[0] = [REMOTE_MODEL_ID]
    values[1] = "An example."
    endpoint = next(index for index, component in enumerate(compare.inputs)
                    if getattr(component, "label", None) == "Remote server base URL")
    values[endpoint] = "http://127.0.0.1:8000"
    asyncio.run(ui.run(compare, *values))
    assert sent == [(REMOTE_MODEL_ID, selected)]
