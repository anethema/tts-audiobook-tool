from dataclasses import replace
import json
import logging

import pytest
import requests

from tts_audiobook_tool import web_backend, web_dialogue
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.l import L
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.combined_analysis import parse_chunk_reply
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil


def _group(text, reason, *, dialogue=False):
    return PhraseGroup([Phrase(text, reason)], is_dialog=dialogue)


def _section(marker):
    return BookSection([
        _group(marker + " " + "word " * 3200, Reason.PARAGRAPH),
        _group("“Come with me.” ", Reason.SECTION_BREAK, dialogue=True),
    ])


def _reply(speakers, characters=()):
    return {"characters": list(characters), "speakers": speakers, "uncertain": []}


def _response(value):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({
        "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 20},
    }).encode("utf-8")
    return response


def test_combined_chunks_resume_checkpoints_and_reuse_manual_corrections(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION"), _section("SECOND-SECTION"), _section("THIRD-SECTION")])
    session = web_backend.BookSession(book, str(tmp_path))
    anna = {"id": "ANNA", "name": "Anna"}
    responses = iter([
        _response(_reply({"1": "ANNA"}, [anna])),
        _response(_reply({})),
        _response(_reply({"1": "ANNA"})),
        _response(_reply({"1": "OTHER:unidentified female"})),
        _response(_reply({"1": "UNKNOWN"})),
    ])
    prompts = []

    def post(_http, _url, **kwargs):
        messages = kwargs["json"]["messages"]
        assert [message["role"] for message in messages] == ["system", "user"]
        prompts.append(messages[1]["content"])
        return next(responses)

    monkeypatch.setattr(requests.Session, "post", post)
    logs = []
    partial = web_backend.analyze_book_dialogue(session, "test-key", on_log=logs.append)
    assert "1 of 3" in partial.attribution_notice
    assert len(prompts) == 2
    assert "120 input / 20 output tokens" in "".join(logs)
    assert "2/3 stopped" in "".join(logs)

    # Recover both roster and labels even if the interrupted browser did not
    # receive the returned session.
    complete = web_backend.analyze_book_dialogue(session, "test-key", on_log=logs.append)
    assert "3 of 3" in complete.attribution_notice
    assert len(prompts) == 4
    assert [marker in prompts[0] for marker in ("FIRST-SECTION", "SECOND-SECTION", "THIRD-SECTION")] == [True, False, False]
    assert "FIRST-SECTION" not in prompts[2]
    assert 'Established characters: [{"id":"ANNA","name":"Anna"}]' in prompts[2]
    assert "1/3 reused saved speaker labels for 1 quote" in "".join(logs)

    corrected = web_dialogue.correct_speaker(complete, 0, 1, "NOT_SPEECH")
    cached = web_backend.analyze_book_dialogue(corrected, "", on_log=logs.append)
    assert len(prompts) == 4
    assert cached.attributions.speaker_of_each_line(book)[1] == "NOT_SPEECH"

    edited_book = replace(book, sections=[book.sections[0], _section("REVISED-SECOND"), book.sections[2]])
    edited = web_backend.analyze_book_dialogue(replace(cached, book=edited_book), "test-key", on_log=logs.append)
    assert len(prompts) == 5
    assert "REVISED-SECOND" in prompts[-1]
    assert edited.attributions.speaker_of_each_line(edited_book) == {
        1: "NOT_SPEECH", 3: "UNKNOWN", 5: "OTHER:unidentified female",
    }
    assert edited.attributions.by_first_line(edited_book)[3].uncertain
    assert web_backend.book_cast_problem(edited) is None
    assert "test-key" not in "".join(logs)


def test_combined_analysis_only_requests_missing_quotes_inside_a_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[BookSection([
        _group("“First.” ", Reason.PARAGRAPH, dialogue=True),
        _group("“Second.” ", Reason.PARAGRAPH, dialogue=True),
    ])])
    existing = RangeAttribution(0, 1, UtteranceUtil.range_text_hash(book, 0, 1), (
        UtteranceAttribution((0,), "ANNA"),
    ))
    session = web_backend.BookSession(
        book, str(tmp_path), roster=Roster((RosterCharacter("ANNA", "Anna"),)),
        attributions=AttributionsFile((existing,)),
    )

    def post(_http, _url, **kwargs):
        passage = kwargs["json"]["messages"][1]["content"]
        assert '<q speaker="ANNA">“First.”</q>' in passage
        assert '<q id="1">' not in passage
        assert '<q id="2">“Second.”</q>' in passage
        return _response(_reply({"2": "NOT_SPEECH"}))

    monkeypatch.setattr(requests.Session, "post", post)
    result = web_backend.analyze_book_dialogue(session, "test-key")
    assert result.attributions.speaker_of_each_line(book) == {0: "ANNA", 1: "NOT_SPEECH"}


def test_combined_reply_alias_deltas_keep_existing_ids_and_distinct_titles():
    roster = Roster((RosterCharacter("ANNA", "Anna", aliases=("the captain",)),))
    additions = [
        {"id": "ANNA_MAY", "name": "Anna May", "aliases": ["Anna"]},
        {"id": "JO", "name": "Jo", "aliases": ["the captain"]},
    ]
    updated, parsed = parse_chunk_reply(json.dumps(_reply({"1": "ANNA_MAY", "2": "JO"}, additions)), roster, {1, 2})
    assert updated.ids == {"ANNA", "JO"}
    assert parsed.speakers == {1: "ANNA", 2: "JO"}
    assert updated.characters[0].aliases == ("the captain", "Anna May")


def test_combined_reply_resolves_persisted_merged_ids():
    current = RosterCharacter("ANNA", "Anna", aliases=("Annah",), merged_ids=("ANNAH", "ANA"))
    roster = Roster.from_dict(Roster((current,)).to_dict())
    assert roster.characters == (current,)

    updated, parsed = parse_chunk_reply(json.dumps(_reply({"1": "ANNAH"})), roster, {1})
    assert updated == roster
    assert parsed.speakers == {1: "ANNA"}

    additions = [
        {"id": "ANNAH", "name": "Annah", "aliases": ["Captain Anna"], "merged_ids": ["JO"]},
        {"id": "JO", "name": "Jo", "merged_ids": ["ANNA"]},
    ]
    updated, parsed = parse_chunk_reply(json.dumps(_reply({"1": "ANNAH", "2": "ANA", "3": "JO"}, additions)),
                                        roster, {1, 2, 3})
    assert updated.ids == {"ANNA", "JO"}
    assert updated.get("ANNA").aliases == ("Annah", "Captain Anna")
    assert updated.get("ANNA").merged_ids == ("ANNAH", "ANA")
    assert updated.get("JO").merged_ids == ()
    assert parsed.speakers == {1: "ANNA", 2: "ANNA", 3: "JO"}


@pytest.mark.parametrize("speakers", ['{"1":"UNKNOWN","1":"NOT_SPEECH"}', '{"1":"INVENTED_ID"}', '{"1":null}'])
def test_combined_reply_rejects_ambiguous_quote_labels(speakers):
    with pytest.raises(ValueError):
        parse_chunk_reply('{"characters":[],"speakers":' + speakers + '}', Roster(()), {1})
