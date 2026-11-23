from dataclasses import replace
from pathlib import Path

import pytest
import requests

from tts_audiobook_tool import web_backend, web_projects
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.web_character_merge import merge_characters, suggest_duplicate_characters
from tts_audiobook_tool.web_render import RenderVoice, _cast_book


def _session(directory):
    book = Book(sections=[BookSection([
        PhraseGroup([Phrase("“Stand by.” ", Reason.SENTENCE)], is_dialog=True),
        PhraseGroup([Phrase("“Now.” ", Reason.SENTENCE)], is_dialog=True, voice_index=1),
        PhraseGroup([Phrase("“Understood.”", Reason.SECTION_BREAK)], is_dialog=True),
    ])])
    roster = Roster((
        RosterCharacter("GRANT", "Grant", is_narrator=True),
        RosterCharacter("SYKORA", "Sykora", aliases=("The captain",)),
        RosterCharacter("SYKORRA", "Sykorra", aliases=("Skori",), merged_ids=("SYKKORA",)),
    ))
    labels = (
        UtteranceAttribution((0,), "SYKORRA", uncertain=True),
        UtteranceAttribution((1,), "SYKORRA"),
        UtteranceAttribution((2,), "GRANT"),
    )
    attributions = AttributionsFile((RangeAttribution(
        0, 3, UtteranceUtil.range_text_hash(book, 0, 3), labels,
    ),))
    return web_backend.BookSession(
        book, str(directory), roster=roster, attributions=attributions,
        line_voices={1: RenderVoice(name="Line override")},
    )


def test_suggest_likely_spelling_variants_and_keep_narrator_first(tmp_path):
    session = replace(_session(tmp_path), roster=Roster((
        RosterCharacter("SYKORRA", "Sykorra"),
        RosterCharacter("SYKORA", "Sykora"),
        RosterCharacter("ALICE", "Alice"),
        RosterCharacter("ALCIE", "Alcie", is_narrator=True),
        RosterCharacter("NORA", "Nora", aliases=("The guard",)),
        RosterCharacter("VERA", "Vera", aliases=("The guard",)),
    )))
    assert suggest_duplicate_characters(session) == [("SYKORRA", "SYKORA"), ("ALCIE", "ALICE")]


def test_merge_remaps_dialogue_and_round_trips_project(tmp_path, monkeypatch):
    session = _session(tmp_path)
    kept_voice = RenderVoice(name="Sykora voice")
    removed_voice = RenderVoice(name="Other voice")
    narrator = RenderVoice(name="Narrator")
    voices = {"SYKORA": kept_voice, "SYKORRA": removed_voice}
    root = tmp_path / "projects"
    project = web_projects.create_project(
        "Merge test", session, narrator=narrator, voices=voices, model_id="venv-breeze", root=root,
    )
    opened = web_projects.load_project(project.project_id, root=root)
    try:
        merged, cast = merge_characters(opened.session, opened.voices, "SYKORA", "SYKORRA")
        assert cast == {"SYKORA": kept_voice}
        assert opened.voices == voices
        assert merged.roster.narrator.id == "GRANT"
        assert merged.roster.get("SYKORRA") is None
        assert merged.roster.get("SYKORA").aliases == ("The captain", "Sykorra", "Skori")
        assert merged.roster.get("SYKORA").merged_ids == ("SYKORRA", "SYKKORA")
        assert merged.attributions.ranges[0].utterances[0].uncertain

        def request(*args, **kwargs):
            pytest.fail("A completed local merge must not send another request.")

        monkeypatch.setattr(requests.Session, "post", request)
        recovered = web_backend.analyze_book_dialogue(opened.session, "")
        assert recovered.roster == merged.roster
        assert recovered.attributions == merged.attributions
        web_projects.save_project(
            project.project_id, merged, narrator=narrator, voices=cast, model_id="venv-breeze", root=root,
        )
    finally:
        web_backend.close_book_session(opened.session)

    reopened = web_projects.load_project(project.project_id, root=root)
    try:
        assert reopened.session.roster == merged.roster
        assert reopened.session.attributions.speaker_of_each_line(reopened.session.book) == {
            0: "SYKORA", 1: "SYKORA", 2: "GRANT",
        }
        rendered, references = _cast_book(reopened.session, {**reopened.voices, "NARRATOR": reopened.narrator})
        assert [references[group.voice_index].name for group in rendered.phrase_groups] == [
            "Sykora voice", "Line override", "Narrator",
        ]
    finally:
        web_backend.close_book_session(reopened.session)


def test_merge_uses_the_keepers_existing_narrator_fallback(tmp_path):
    session = _session(tmp_path)
    merged, cast = merge_characters(
        session, {"SYKORRA": RenderVoice(name="Source voice")}, "SYKORA", "SYKORRA",
    )
    assert cast == {}
    narrator = RenderVoice(name="Narrator")
    rendered, references = _cast_book(merged, {"NARRATOR": narrator})
    assert references[rendered.phrase_groups[0].voice_index] == narrator


def test_invalid_merge_leaves_session_and_speaker_file_intact(tmp_path):
    session = _session(tmp_path)
    assert not SpeakerAssignFileUtil.save_attributions_file(session.project_dir, session.attributions)
    checkpoint = Path(SpeakerAssignFileUtil.attributions_path(session.project_dir))
    original = checkpoint.read_bytes()
    for keep, remove, message in (
        ("SYKORA", "GRANT", "narrator"),
        ("SYKORA", "SYKORA", "different characters"),
        ("SYKORA", "OTHER", "current cast"),
    ):
        with pytest.raises(web_backend.WebBackendError, match=message):
            merge_characters(session, {}, keep, remove)
        assert checkpoint.read_bytes() == original
