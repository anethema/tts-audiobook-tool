import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from filelock import FileLock
import numpy as np
import pytest
import soundfile

from tts_audiobook_tool import web_book_outputs
from tts_audiobook_tool.app_types import Book, BookSection, ExportType, NormalizationType
from tts_audiobook_tool.app_types.app_metadata import AppMetadata
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil, get_segment_stt_info_path
from tts_audiobook_tool.web_backend import BookSession


RUN_ID = "a" * 64


@pytest.fixture
def rendered_book(tmp_path):
    if not hasattr(L, "logger"):
        L.init("test_web_book_outputs")
    directory = tmp_path / "renders" / RUN_ID
    directory.mkdir(parents=True)
    book = Book([
        BookSection([
            PhraseGroup([Phrase("The original line.", Reason.SENTENCE)]),
            PhraseGroup([Phrase("An unfinished line.", Reason.SECTION_BREAK)]),
        ], title="First chapter"),
        BookSection([PhraseGroup([Phrase("The closing line.", Reason.SECTION_BREAK)])], title="Second chapter"),
    ], title="An audiobook")
    project = Project(dir_path=str(directory), language_code="en", tts_model_type="breeze_tts_2_local")
    project.book = book
    try:
        assert ProjectTextIOUtil.save_book(project) == ""
        assert project.save() == ""
        sr = 24000
        time = np.arange(sr, dtype=np.float32) / sr
        fixture_audio = 0.12 * np.sin(2 * np.pi * 430 * time)
        best = ""
        for index, tags, label in [(0, " [2]", "mistakes"), (0, "", "preferred"), (2, "", "ending")]:
            digest = SoundSegmentUtil.calc_segment_hash(index, book.phrase_groups[index].text)
            filename = f"[{index + 1:05}] [{digest}] [breeze2] [narrator]{tags} {label}.flac"
            path = Path(project.sound_segments_path) / filename
            soundfile.write(path, fixture_audio, sr)
            if label == "preferred":
                best = filename
                get_segment_stt_info_path(path).write_text("{}", encoding="utf-8")
        old = SoundSegmentUtil.calc_segment_hash(1, "Stale text.")
        soundfile.write(Path(project.sound_segments_path) / f"[00002] [{old}] stale.flac", fixture_audio, sr)
    finally:
        project.kill()
        project.sound_segments.observer.join()
    edited = Book([BookSection([PhraseGroup([Phrase("The changed line.", Reason.SECTION_BREAK)])])])
    return BookSession(edited, str(tmp_path)), directory, best


def test_render_review_uses_stored_text_and_native_take_ranking(rendered_book):
    session, directory, best = rendered_book
    (directory.parent / ("b" * 64)).mkdir()
    (directory / "audiobook.abr.m4b").write_bytes(b"existing full output")
    runs = web_book_outputs.list_book_render_runs(session)
    assert [(run.run_id, run.title, run.generated_lines, run.total_lines) for run in runs] == [
        (RUN_ID, "An audiobook", 2, 3),
    ]
    review = web_book_outputs.review_book_line(session, RUN_ID, 1)
    assert review.text == "The original line."
    assert [(take.preferred, take.word_errors) for take in review.takes] == [(True, 0), (False, 2)]
    assert not web_book_outputs.review_book_line(session, RUN_ID, 2).takes
    path = web_book_outputs.get_book_take_path(session, RUN_ID, 1, best)
    assert soundfile.info(path).duration == 1.0
    assert web_book_outputs.list_book_exports(session, RUN_ID) == (("audiobook.abr.m4b", "audiobook.abr.m4b"),)
    assert web_book_outputs.get_book_export_path(session, RUN_ID, "audiobook.abr.m4b") == str(
        directory / "audiobook.abr.m4b",
    )

    with pytest.raises(web_book_outputs.WebBookOutputError, match="available take"):
        web_book_outputs.get_book_take_path(session, RUN_ID, 3, best)
    with pytest.raises(web_book_outputs.WebBookOutputError, match="available audiobook"):
        web_book_outputs.get_book_export_path(session, RUN_ID, "../../project_text.json")
    with pytest.raises(web_book_outputs.WebBookOutputError, match="from this project"):
        web_book_outputs.review_book_line(session, "../other", 1)


@pytest.mark.parametrize("container", [ExportType.FLAC, ExportType.AAC])
def test_review_export_uses_native_audio_and_embedded_metadata(rendered_book, container):
    session, directory, _ = rendered_book
    config_path = directory / "project.json"
    before = config_path.read_bytes()
    result = web_book_outputs.export_book_audio(
        session, RUN_ID, first_line=1, last_line=2, bookmarks=(2,),
        options=web_book_outputs.BookExportOptions(
            export_type=container, normalization_type=NormalizationType.STRONGER, use_break_sound_effect=False,
        ),
    )
    assert (result.line_start, result.line_end, result.missing_lines) == (1, 2, 1)
    path = Path(result.path)
    assert path.name.endswith(f".abr{container.suffix}") and path.stat().st_size > 0
    metadata = AppMetadata.load_from_file(str(path))
    assert metadata is not None
    assert metadata.title == "An audiobook"
    assert metadata.bookmark_indices == [1]
    assert metadata.project_snapshot["export_type"] == container.id
    assert config_path.read_bytes() == before
    choices = web_book_outputs.list_book_exports(session, RUN_ID)
    assert len(choices) == 1
    assert web_book_outputs.get_book_export_path(session, RUN_ID, choices[0][1]) == result.path


def test_export_requires_a_take_and_respects_the_render_lock(rendered_book):
    session, directory, _ = rendered_book
    with pytest.raises(web_book_outputs.WebBookOutputError, match="no generated audio"):
        web_book_outputs.export_book_audio(session, RUN_ID, first_line=2, last_line=2)
    with FileLock(str(directory / ".render.lock"), timeout=0):
        with pytest.raises(web_book_outputs.WebBookOutputError, match="already generating or exporting"):
            web_book_outputs.export_book_audio(session, RUN_ID)


def test_split_exports_keep_per_file_audio_and_native_reader_metadata(rendered_book):
    session, directory, _ = rendered_book
    config = directory / "project.json"
    before = config.read_bytes()
    with web_book_outputs._open_render_project(directory) as render:
        repaired = deepcopy(render.book)
    repaired.phrase_groups[2].phrases[0].fmt = "The <i>closing</i> line."
    session = replace(session, book=repaired)
    outputs = web_book_outputs.export_book_audio_parts(
        session, RUN_ID, cut_lines=(1, 3, 3), split_at_sections=True,
        options=web_book_outputs.BookExportOptions(export_type=ExportType.AAC, use_break_sound_effect=False),
    )
    assert [(part.line_start, part.line_end, part.missing_lines) for part in outputs] == [(1, 2, 1), (3, 3, 0)]
    assert len({Path(part.path).parent for part in outputs}) == 1
    assert config.read_bytes() == before
    assert {part.export_id for part in outputs} == {export_id for _, export_id in web_book_outputs.list_book_exports(session, RUN_ID)}
    for index, part in zip((0, 2), outputs, strict=True):
        assert web_book_outputs.get_book_export_path(session, RUN_ID, part.export_id) == part.path
        metadata = AppMetadata.load_from_file(part.path)
        assert metadata is not None
        assert metadata.bookmark_indices == []
        assert metadata.project_snapshot["markers"] == [2]
        assert [section.title for section in metadata.sections] == ["First chapter", "Second chapter"]
        assert [i for i, phrase in enumerate(metadata.timed_phrases) if phrase.time_end > phrase.time_start] == [index]
        assert metadata.timed_phrases[index].time_start == 0
        assert metadata.timed_phrases[2].fmt == "The <i>closing</i> line."


def test_split_validation_runs_before_writing_and_errors_do_not_publish_parts(rendered_book, monkeypatch):
    session, directory, _ = rendered_book
    calls = []

    def make_file(_state, start, end, _marks, stem, _upsample):
        calls.append((start, end))
        if start:
            return "", "Simulated encoding failure"
        output = Path(stem + ".abr.m4b")
        output.write_bytes(b"First part")
        return str(output), ""

    monkeypatch.setattr(web_book_outputs, "_make_audio_file", make_file)
    with pytest.raises(web_book_outputs.WebBookOutputError, match="Bookmarks are available"):
        web_book_outputs.export_book_audio_parts(session, RUN_ID, cut_lines=(3,), bookmarks=(1,))
    with pytest.raises(web_book_outputs.WebBookOutputError, match="Split lines must be within the export range"):
        web_book_outputs.export_book_audio_parts(session, RUN_ID, last_line=2, cut_lines=(3,))
    with pytest.raises(web_book_outputs.WebBookOutputError, match="no generated audio in part lines 2"):
        web_book_outputs.export_book_audio_parts(session, RUN_ID, cut_lines=(2, 3))
    assert calls == []

    with pytest.raises(web_book_outputs.WebBookOutputError, match="could not be exported"):
        web_book_outputs.export_book_audio_parts(session, RUN_ID, cut_lines=(3,))
    assert calls == [(0, 1), (2, 2)]
    assert web_book_outputs.list_book_exports(session, RUN_ID) == ()
    assert not tuple(directory.glob(".export-*"))


def test_saved_exports_can_open_all_parts_of_a_split(rendered_book):
    session, directory, _ = rendered_book
    with web_book_outputs._open_render_project(directory) as project:
        export_dir = Path(project.concat_path) / "export-twelve"
    export_dir.mkdir(parents=True)
    files = [export_dir / f"Book [{part} of 12].abr.flac" for part in range(1, 13)]
    for file in files:
        file.write_bytes(b"Saved audio")
    choices = web_book_outputs.list_book_exports(session, RUN_ID)
    assert len(choices) == 12
    assert [web_book_outputs.get_book_export_path(session, RUN_ID, export_id) for _, export_id in choices] == list(map(str, files))


def test_panel_opens_reader_for_each_split_part(rendered_book, monkeypatch, tmp_path):
    import gradio as gr
    from gradio.state_holder import SessionState

    from tts_audiobook_tool import web_book_output_panel

    session, directory, _ = rendered_book
    with web_book_outputs._open_render_project(directory) as project:
        export_dir = Path(project.concat_path) / "export-two"
    export_dir.mkdir(parents=True)
    outputs = []
    for number, (first, last) in enumerate(((1, 2), (3, 3)), start=1):
        path = export_dir / f"Part {number}.abr.m4b"
        path.write_bytes(f"Audio for part {number}".encode())
        outputs.append(web_book_outputs.BookExport(str(path), first, last, 0, path.relative_to(directory).as_posix()))

    def fake_export(session_arg, run_id, **options):
        assert session_arg is session
        assert run_id == RUN_ID
        assert options["cut_lines"] == (3,)
        assert options["bookmarks"] is None
        return tuple(outputs)

    async def exercise():
        with gr.Blocks(analytics_enabled=False) as app:
            book = gr.State(None)
            web_book_output_panel.create_book_output_panel(book, private_event={"api_visibility": "private"})
        app.has_launched = True
        state = SessionState(app)
        state[book._id] = session
        request = gr.Request(session_hash="test-split-reader")

        def find(name):
            return next(fn for fn in app.fns.values() if fn.name == name)

        await app.process_api(find("find_renderings"), [None, None], state, request=request)
        export = find("make_export")
        supplied = {"Rendering": RUN_ID, "Split before lines": "3"}
        values = [supplied.get(getattr(component, "label", None), component.value) for component in export.inputs]
        done = (await app.process_api(export, values, state, request=request))["data"]
        assert done[2]["value"] == outputs[0].export_id
        assert done[0]["value"]["orig_name"] == "Part 1.abr.m4b"
        assert "/reader/?" in done[1]["value"] and "Part%25201.abr.m4b" in done[1]["value"]

        opened = await app.process_api(find("open_export"), [None, RUN_ID, outputs[1].export_id], state, request=request)
        download, reader, _ = opened["data"]
        assert download["value"]["orig_name"] == "Part 2.abr.m4b"
        assert "/reader/?" in reader["value"] and "Part%25202.abr.m4b" in reader["value"]

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "downloads"))
    monkeypatch.setattr(web_book_output_panel, "export_book_audio_parts", fake_export)
    asyncio.run(exercise())


def test_panel_registers_review_and_export_callbacks():
    import gradio as gr

    from tts_audiobook_tool.web_book_output_panel import create_book_output_panel

    with gr.Blocks() as app:
        create_book_output_panel(gr.State(None), private_event={"api_visibility": "private"})
    callbacks = {dependency["api_name"] for dependency in app.config["dependencies"]}
    assert {"find_renderings", "read_line", "play_take", "make_export", "open_export"} <= callbacks
    labels = {component["props"].get("label") for component in app.config["components"]}
    assert {"Bookmark lines", "Split before lines", "Split at saved section boundaries and file split points"} <= labels
