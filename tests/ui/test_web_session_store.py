from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from tts_audiobook_tool import web_backend, web_projects, web_session_store as drafts
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.web_render import RenderVoice


def test_private_draft_survives_workspace_deletion_and_conflicting_tabs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    token = "a" * 32
    book = Book(sections=[BookSection([PhraseGroup([Phrase("Hello", Reason.SECTION_BREAK)])])], title="Book")
    session = drafts.bind_draft_workspace(token, web_backend.new_book_session(book))
    epub = tmp_path / "source.epub"
    epub.write_bytes(b"example")
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"audio")
    fields = dict(
        active_project_id=None, narrator=RenderVoice(str(voice), "transcript"), voices={}, model_id="breeze",
        project_name="Editing", source_epub=str(epub), performance_cues=True,
        controls={"use_local_qwen": False, "reasoning_effort": "low", "api_key": "test-secret",
                  "llm_endpoint": "https://example.invalid/chat?token=test-secret",
                  "llm_extra_params": '{"authorization":"test-secret"}'},
    )
    revision = drafts.save_draft(token, None, session, **fields)
    assert drafts.load_draft("b" * 32) is None
    saved = drafts.load_draft(token)
    assert saved.project.session.analysis_scope is None
    assert saved.omitted_request_settings
    assert saved.revision == revision and saved.project.summary.name == "Editing"
    assert saved.controls == {"use_local_qwen": False, "reasoning_effort": "low", "llm_endpoint": "",
                              "llm_extra_params": ""}
    assert saved.project.session.project_dir == session.project_dir
    (Path(session.project_dir) / "audio-kept").touch()
    # This is the typical setting-change path for a 50k-line book. Keep the book on disk.
    fields["controls"] = {"reasoning_effort": "medium", "llm_endpoint": "https://api.openai.com/v1/chat/completions",
                          "llm_extra_params": '{"temperature":0.2}'}
    with patch.object(web_projects, "book_to_project_text_json_dict", side_effect=AssertionError("rewrote book")):
        rev2 = drafts.save_draft(token, revision, session, **fields)
    with pytest.raises(web_projects.WebProjectError, match="another tab"):
        drafts.save_draft(token, revision, saved.project.session, **fields)
    web_backend.close_book_session(session)
    web_backend.close_book_session(saved.project.session)
    epub.unlink()
    voice.unlink()
    reopened = drafts.load_draft(token)
    assert reopened.revision == rev2 and reopened.controls == fields["controls"]
    assert not reopened.omitted_request_settings
    assert Path(reopened.project.source_epub).read_bytes() == b"example"
    assert Path(reopened.project.narrator.reference_path).read_bytes() == b"audio"
    assert (Path(reopened.project.session.project_dir) / "audio-kept").exists()
    directory = drafts.draft_root() / token
    assert not directory.stat().st_mode & 0o077
    for f in directory.glob("*.json"):
        assert not f.stat().st_mode & 0o077 and b"test-secret" not in f.read_bytes()


def test_named_project_has_independent_draft_and_up_to_date_analysis_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    book = Book(sections=[BookSection([PhraseGroup([Phrase("“Hello.”", Reason.SECTION_BREAK)], is_dialog=True)])],
                title="Book")
    session = web_backend.new_book_session(book)
    fields = dict(narrator=RenderVoice(), voices={}, model_id="breeze")
    named = web_projects.create_project("Named", session, **fields)
    session = web_projects.load_project(named.project_id).session
    attrs = AttributionsFile((RangeAttribution(0, 1, UtteranceUtil.range_text_hash(book, 0, 1),
                             (UtteranceAttribution((0,), "UNKNOWN"),)),))
    from tts_audiobook_tool.speaker_assign.roster import Roster
    session = replace(session, roster=Roster(()), attributions=attrs, attribution_notice="1 saved")
    path = Path(web_backend.SpeakerAssignFileUtil.attributions_path(session.attribution_dir))
    origin = dict(book="a" * 64, config="b" * 64, baseline="c" * 64)
    web_backend._save_analysis_checkpoint(path, session.roster, attrs, origin=origin)
    token = "0" * 32
    revision = drafts.save_draft(token, None, session, active_project_id=named.project_id, controls={},
                                 project_name="Draft name", **fields)
    web_backend.close_book_session(session)
    opened = drafts.load_draft(token)
    assert opened.active_project_id == named.project_id and opened.analysis_origin == origin
    assert opened.project.session.analysis_scope == session.project_dir
    assert opened.project.session.project_dir == session.project_dir
    assert opened.project.session.attributions == attrs
    assert not web_projects.load_project(named.project_id).session.attributions.ranges
    # An uncommitted replacement must leave the previous entire revision readable.
    edited = replace(opened.project.session, attribution_notice="updated")
    original_save = web_projects.JsonSaveUtil.save

    def fail_manifest(kind, path, factory):
        return True if Path(path).name == "session.json" else original_save(kind, path, factory)

    with patch.object(web_projects.JsonSaveUtil, "save", side_effect=fail_manifest):
        with pytest.raises(web_projects.WebProjectError, match="browser session"):
            drafts.save_draft(token, revision, edited, active_project_id=named.project_id, controls={}, **fields)
    assert drafts.load_draft(token).project.session.attribution_notice == "1 saved"
    with pytest.raises(web_projects.WebProjectError, match="identifier"):
        drafts.load_draft("../" + token)


def test_imported_books_use_separate_render_runs_within_one_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    token = "b" * 32
    def make(title):
        return web_backend.new_book_session(Book(sections=[], title=title))

    first = drafts.bind_draft_workspace(token, make("First"))
    first_run = Path(first.project_dir) / "renders" / ("a" * 64)
    first_run.mkdir(parents=True)
    for filename in (web_projects.PROJECT_TEXT_FILE_NAME, web_projects.PROJECT_JSON_FILE_NAME):
        (first_run / filename).write_text("{}")

    second = drafts.bind_draft_workspace(token, make("Second"))
    second_again = drafts.bind_draft_workspace(token, make("Second"))
    assert first.project_dir != second.project_dir == second_again.project_dir
    fields = dict(narrator=RenderVoice(), voices={}, model_id="")
    drafts.save_draft(token, None, second, active_project_id=None, controls={}, **fields)
    restored = drafts.load_draft(token)
    assert restored.project.session.project_dir == second.project_dir
    saved = web_projects.create_project("Second", restored.project.session, **fields)
    assert not (Path(web_projects.load_project(saved.project_id).session.project_dir) / "renders").exists()
