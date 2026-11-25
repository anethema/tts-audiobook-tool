from dataclasses import replace
import json
import logging

import pytest
import requests

from tts_audiobook_tool import web_backend, web_dialogue
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.delivery_cue import DeliveryCue
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.l import L
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.combined_analysis import PERFORMANCE_CUE_SYSTEM_PROMPT, parse_chunk_reply
from tts_audiobook_tool.speaker_assign.roster import (
    CHARACTER_VOICE_DIRECTIONS, NEUTRAL_CHARACTER_VOICE_DIRECTION, Roster, RosterCharacter,
)
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
    return _content_response(json.dumps(value))


def _content_response(content, *, finish_reason="stop"):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
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
        _response(_reply({}, [{"name": "RAW-ROSTER-FIELD"}])),
        _response(_reply({})),
        _response(_reply({"1": "UNLISTED_SPEAKER"})),
        _response(_reply({"1": "OTHER:unidentified female"})),
        _response(_reply({"1": "ANNA"})),
        _response(_reply({"1": "UNKNOWN"})),
    ])
    prompts = []
    progress = []
    progress_at_request = []

    def track(done, total, detail):
        progress.append((done, total, detail))

    def post(_http, _url, **kwargs):
        messages = kwargs["json"]["messages"]
        assert [message["role"] for message in messages] == ["system", "user"]
        prompts.append(messages)
        progress_at_request.append(progress[-1])
        return next(responses)

    monkeypatch.setattr(requests.Session, "post", post)
    logs = []
    partial = web_backend.analyze_book_dialogue(session, "test-key", on_log=logs.append, on_progress=track)
    assert [done for done, _, _ in progress_at_request] == [0, 1, 1, 1, 1]
    assert all(total == 3 for _, total, _ in progress)
    assert "Chunk 2/3 (lines 3–4)" in progress_at_request[1][2]
    assert "retry 1/2" in progress_at_request[2][2]
    assert "retry 2/2" in progress_at_request[3][2]
    assert any(done == 1 and "skipped" in detail for done, _, detail in progress)
    assert progress[-1][:2] == (2, 3)
    assert "2 of 3" in partial.attribution_notice
    assert "Skipped chunk 2 (lines 3–4)" in partial.attribution_notice
    assert "Run Analyze book to retry" in partial.attribution_notice
    assert len(prompts) == 5
    assert len({messages[1]["content"] for messages in prompts[1:4]}) == 1
    assert "Current passage, starting at line 5:" in prompts[4][1]["content"]
    assert "Retry reminder:" not in prompts[4][0]["content"]
    assert "RAW-ROSTER-FIELD" not in repr(prompts)
    assert "RAW-ROSTER-FIELD" not in "".join(logs)
    assert "Retrying (1/2)" in "".join(logs)
    assert "Retrying (2/2)" in "".join(logs)
    assert "120 input / 20 output tokens" in "".join(logs)
    assert "Chunk 2/3 skipped (lines 3–4) after 3 invalid replies." in "".join(logs)
    assert "speaker outside the character list." in "".join(logs)
    assert "Run Analyze book to retry" in "".join(logs)
    assert [saved.key for saved in partial.attributions.ranges] == [(0, 2), (4, 6)]
    assert partial.attributions.speaker_of_each_line(book) == {1: "ANNA", 5: "OTHER:unidentified female"}
    assert partial.roster.ids == {"ANNA"}
    assert web_backend.book_cast_problem(partial, first_line=4, last_line=4)
    assert web_backend.book_cast_problem(partial, first_line=5, last_line=6) is None

    # Recover both roster and labels even if the browser did not
    # receive the returned session.
    progress.clear()
    complete = web_backend.analyze_book_dialogue(session, "test-key", on_log=logs.append, on_progress=track)
    assert progress[0][:2] == (2, 3)
    assert "Reusing" in progress[0][2]
    assert progress_at_request[-1][:2] == (2, 3)
    assert "Chunk 2/3 (lines 3–4)" in progress_at_request[-1][2]
    assert progress[-1][:2] == (3, 3)
    assert "3 of 3" in complete.attribution_notice
    assert "Skipped" not in complete.attribution_notice
    assert len(prompts) == 6
    first_passage = prompts[0][1]["content"]
    assert [marker in first_passage for marker in ("FIRST-SECTION", "SECOND-SECTION", "THIRD-SECTION")] == [True, False, False]
    resumed_passage = prompts[5][1]["content"]
    assert "FIRST-SECTION" not in resumed_passage
    assert "Current passage, starting at line 3:" in resumed_passage
    assert 'Established characters: [{"id":"ANNA","name":"Anna"}]' in resumed_passage
    assert "1/3 reused saved speaker labels for 1 quote" in "".join(logs)
    assert "3/3 reused saved speaker labels for 1 quote" in "".join(logs)

    corrected = web_dialogue.correct_speaker(complete, 0, 1, "NOT_SPEECH")
    # The correction is recoverable with the roster even from the original browser state.
    progress.clear()
    cached = web_backend.analyze_book_dialogue(session, "", on_log=logs.append, on_progress=track)
    assert progress[0][:2] == (3, 3)
    assert "Reusing" in progress[0][2]
    assert cached.roster == corrected.roster
    assert len(prompts) == 6
    assert cached.attributions.speaker_of_each_line(book)[1] == "NOT_SPEECH"

    edited_book = replace(book, sections=[book.sections[0], _section("REVISED-SECOND"), book.sections[2]])
    edited = web_backend.analyze_book_dialogue(
        replace(cached, book=edited_book), "test-key", on_log=logs.append, on_progress=track,
    )
    assert len(prompts) == 7
    assert "REVISED-SECOND" in prompts[-1][1]["content"]
    assert edited.attributions.speaker_of_each_line(edited_book) == {
        1: "NOT_SPEECH", 3: "UNKNOWN", 5: "OTHER:unidentified female",
    }
    assert edited.attributions.by_first_line(edited_book)[3].uncertain
    assert web_backend.book_cast_problem(edited) is None
    assert "test-key" not in "".join(logs)


def test_combined_analysis_corrects_invalid_replies_with_a_fresh_retry_budget_per_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION"), _section("SECOND-SECTION")])
    session = web_backend.BookSession(book, str(tmp_path))
    responses = iter([
        _content_response('{"RAW-REPLY-TEXT":'),
        _response(_reply({"1": "UNLISTED_SPEAKER"})),
        _response(_reply({"1": "UNKNOWN"})),
        _response(_reply({})),
        _response(_reply({"1": "NOT_SPEECH"})),
    ])
    requests_sent = []
    logs = []

    def post(_http, _url, **kwargs):
        messages = kwargs["json"]["messages"]
        assert [message["role"] for message in messages] == ["system", "user"]
        requests_sent.append([message["content"] for message in messages])
        return next(responses)

    monkeypatch.setattr(requests.Session, "post", post)
    result = web_backend.analyze_book_dialogue(session, "test-key", on_log=logs.append)
    assert len(requests_sent) == 5
    assert len({passage for _, passage in requests_sent[:3]}) == 1
    assert requests_sent[3][1] == requests_sent[4][1]
    assert requests_sent[0][0] == requests_sent[3][0]
    assert "could not be parsed as JSON" in requests_sent[1][0]
    assert "speaker outside the character list" in requests_sent[2][0]
    assert "exactly the requested quote IDs" in requests_sent[4][0]
    assert all(system.count("Retry reminder:") == 1 for system, _ in (
        requests_sent[1], requests_sent[2], requests_sent[4],
    ))
    assert "2 of 2" in result.attribution_notice
    assert result.attributions.speaker_of_each_line(book) == {1: "UNKNOWN", 3: "NOT_SPEECH"}
    messages = "".join(logs)
    assert "Chunk 1/2 The reply could not be parsed as JSON. Retrying (1/2)." in messages
    assert "Chunk 1/2 The reply named a speaker outside the character list. Retrying (2/2)." in messages
    assert "Chunk 2/2 The reply did not contain exactly the requested quote IDs. Retrying (1/2)." in messages
    for raw_text in ("RAW-REPLY-TEXT", "UNLISTED_SPEAKER", "test-key"):
        assert raw_text not in repr(requests_sent)
        assert raw_text not in messages


def test_all_invalid_chunks_report_the_gaps_without_saving_assignments(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION"), _section("SECOND-SECTION")])
    session = web_backend.BookSession(book, str(tmp_path))
    prompts = []
    logs = []

    def post(_http, _url, **kwargs):
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return _content_response("No valid JSON")

    monkeypatch.setattr(requests.Session, "post", post)
    result = web_backend.analyze_book_dialogue(session, "test-key", on_log=logs.append)
    assert len(prompts) == 6
    assert all("Current passage, starting at line 1:" in prompt for prompt in prompts[:3])
    assert all("Current passage, starting at line 3:" in prompt for prompt in prompts[3:])
    assert "0 of 2" in result.attribution_notice
    assert "Skipped chunks 1 (lines 1–2), 2 (lines 3–4)" in result.attribution_notice
    assert "Chunk 2/2 skipped (lines 3–4)" in "".join(logs)
    assert result.roster is not None
    assert not result.attributions.ranges
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("status", "finish_reason", "message"),
    [(429, "stop", "rate-limited"), (200, "length", "output limit")],
)
def test_combined_analysis_does_not_retry_provider_or_output_limit_errors(
        tmp_path, monkeypatch, status, finish_reason, message,
):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION"), _section("SECOND-SECTION")])
    session = web_backend.BookSession(book, str(tmp_path))
    response = _content_response('{"unfinished":', finish_reason=finish_reason)
    response.status_code = status
    calls = []
    logs = []

    def post(_http, _url, **kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(web_backend.WebBackendError, match=message):
        web_backend.analyze_book_dialogue(session, "test-key", on_log=logs.append)
    assert len(calls) == 1
    assert "Retrying (" not in "".join(logs)
    assert "skipped" not in "".join(logs)
    assert not list(tmp_path.iterdir())


def test_provider_error_stops_later_chunks_but_keeps_earlier_gaps_in_status(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION"), _section("SECOND-SECTION"), _section("THIRD-SECTION")])
    session = web_backend.BookSession(book, str(tmp_path))
    invalid = _response(_reply({}))
    rate_limited = _content_response("")
    rate_limited.status_code = 429
    responses = iter([invalid, invalid, invalid, rate_limited])
    prompts = []
    logs = []

    def post(_http, _url, **kwargs):
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return next(responses)

    monkeypatch.setattr(requests.Session, "post", post)
    result = web_backend.analyze_book_dialogue(session, "test-key", on_log=logs.append)
    assert len(prompts) == 4
    assert "Current passage, starting at line 3:" in prompts[-1]
    assert "0 of 3" in result.attribution_notice
    assert "Skipped chunk 1 (lines 1–2)" in result.attribution_notice
    assert "rate-limited" in result.attribution_notice
    assert "Chunk 2/3 stopped." in "".join(logs)
    assert not result.attributions.ranges


def test_combined_analysis_only_requests_missing_quotes_inside_a_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[BookSection([
        _group("“First.” ", Reason.PARAGRAPH, dialogue=True),
        _group("“Second.” ", Reason.PARAGRAPH, dialogue=True),
    ])])
    # A matching saved range alone cannot certify that both quotes were tagged.
    existing = RangeAttribution(0, 2, UtteranceUtil.range_text_hash(book, 0, 2), (
        UtteranceAttribution((0,), "ANNA"),
    ))
    session = web_backend.BookSession(
        book, str(tmp_path), roster=Roster((RosterCharacter("ANNA", "Anna"),)),
        attributions=AttributionsFile((existing,)),
    )
    assert web_backend.book_cast_problem(session, first_line=1, last_line=1) is None
    assert web_backend.book_cast_problem(session, first_line=2, last_line=2)
    assert not web_dialogue.dialogue_sections(session)
    invalid = _response(_reply({}))
    responses = iter([invalid, invalid, invalid, _response(_reply({"2": "NOT_SPEECH"}))])

    def post(_http, _url, **kwargs):
        passage = kwargs["json"]["messages"][1]["content"]
        assert '<q speaker="ANNA">“First.”</q>' in passage
        assert '<q id="1">' not in passage
        assert '<q id="2">“Second.”</q>' in passage
        return next(responses)

    monkeypatch.setattr(requests.Session, "post", post)
    partial = web_backend.analyze_book_dialogue(session, "test-key")
    assert "Skipped chunk 1 (lines 1–2)" in partial.attribution_notice
    assert partial.attributions.ranges == (existing,)
    reopened = web_backend.new_book_session(book, roster=partial.roster, attributions=partial.attributions)
    try:
        result = web_backend.analyze_book_dialogue(reopened, "test-key")
    finally:
        web_backend.close_book_session(reopened)
    assert result.attributions.speaker_of_each_line(book) == {0: "ANNA", 1: "NOT_SPEECH"}
    assert web_backend.book_cast_problem(result) is None
    assert web_dialogue.dialogue_sections(result)


def test_combined_analysis_adds_delivery_without_replacing_saved_speakers(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[BookSection([
        _group("Anna managed a hollow laugh. ", Reason.SENTENCE),
        _group("“All right.” ", Reason.PARAGRAPH, dialogue=True),
        _group("“Quiet Zone” ", Reason.PARAGRAPH, dialogue=True),
        _group("“Let's go.” ", Reason.PARAGRAPH, dialogue=True),
    ])])
    existing = RangeAttribution(0, 3, UtteranceUtil.range_text_hash(book, 0, 3), (
        UtteranceAttribution((1,), "ANNA", uncertain=True), UtteranceAttribution((2,), "NOT_SPEECH"),
    ))
    session = web_backend.BookSession(
        book, str(tmp_path), roster=Roster((RosterCharacter("ANNA", "Anna"),)),
        attributions=AttributionsFile((existing,)),
    )
    prompts = []

    def post(_http, _url, **kwargs):
        messages = kwargs["json"]["messages"]
        prompts.append(messages)
        return _response(_reply({"3": "NOT_SPEECH"}) | {"delivery": {
            "1": {"direction": "with resignation", "before": "laughs"},
            "2": {"after": "sighs"},
            "3": {"after": "gasps"},
        }})

    monkeypatch.setattr(requests.Session, "post", post)
    work = []
    result = web_backend.analyze_book_dialogue(
        session, "test-key", performance_cues=True, on_work_progress=lambda done, total: work.append((done, total)),
    )
    assert set(work) == {(0, 2), (2, 2)}
    assert len(prompts) == 1
    instructions = prompts[0][0]["content"]
    assert "nearly every spoken quote" in instructions
    assert 'Usually omit "before" and "after"' in instructions
    assert "Keep the quoted words intact" in instructions
    passage = prompts[0][1]["content"]
    assert "Anna managed a hollow laugh." in passage
    assert '<q id="1" speaker="ANNA" cue="true">“All right.”</q>' in passage
    assert '<q speaker="NOT_SPEECH">“Quiet Zone”</q>' in passage
    assert '<q id="3" cue="true">“Let\'s go.”</q>' in passage
    saved = result.attributions.by_first_line(book)
    assert saved[1] == replace(existing.utterances[0], delivery=DeliveryCue("with resignation", before="laughs"))
    assert saved[2].delivery is saved[3].delivery is None

    corrected = web_dialogue.correct_speaker(result, 0, 1, "UNKNOWN")
    cached = web_backend.analyze_book_dialogue(corrected, "", performance_cues=True)
    assert len(prompts) == 1
    assert cached.attributions.by_first_line(book)[1] == replace(saved[1], speaker="UNKNOWN", uncertain=False)
    non_speech = web_dialogue.correct_speaker(cached, 0, 1, "NOT_SPEECH")
    assert non_speech.attributions.by_first_line(book)[1].delivery is None


def test_delivery_enrichment_checkpoints_empty_results_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION"), _section("SECOND-SECTION")])
    existing = AttributionsFile(tuple(
        RangeAttribution(start, end, UtteranceUtil.range_text_hash(book, start, end), (
            UtteranceAttribution((end - 1,), "UNKNOWN"),
        ))
        for start, end in book.section_ranges()
    ))
    session = web_backend.BookSession(book, str(tmp_path), attributions=existing)
    responses = iter([
        _response(_reply({}) | {"delivery": {}}),
        _response(_reply({"1": "UNKNOWN"})),
        _response(_reply({"1": "UNKNOWN"})),
        _response(_reply({"1": "UNKNOWN"})),
        _response(_reply({}) | {"delivery": {"1": {"after": "sighs"}}}),
    ])
    prompts = []

    def post(_http, _url, **kwargs):
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return next(responses)

    monkeypatch.setattr(requests.Session, "post", post)
    work = []

    def track_work(done, total):
        work.append((done, total))

    partial = web_backend.analyze_book_dialogue(session, "test-key", performance_cues=True, on_work_progress=track_work)
    assert set(work) == {(0, 2), (1, 2)}
    assert "1 of 2" in partial.attribution_notice
    assert "Skipped chunk 2 (lines 3–4)" in partial.attribution_notice
    assert "Following context (evidence only):\nSECOND-SECTION" in prompts[0]
    assert partial.attributions.by_first_line(book)[1].delivery == DeliveryCue()
    assert partial.attributions.by_first_line(book)[3].delivery is None
    assert partial.attributions.to_dict()["ranges"][0]["utterances"][0]["delivery"] == {}

    work.clear()
    complete = web_backend.analyze_book_dialogue(session, "test-key", performance_cues=True, on_work_progress=track_work)
    assert work[0] == (0, 1) and work[-1] == (1, 1)
    assert len(prompts) == 5
    assert len(set(prompts[1:])) == 1
    assert "2 of 2" in complete.attribution_notice
    assert complete.attributions.by_first_line(book)[3].delivery == DeliveryCue(after="sighs")


def test_character_voices_share_the_combined_request_and_keep_established_direction(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION"), _section("SECOND-SECTION")])
    session = web_backend.BookSession(book, str(tmp_path))
    direction = "Slightly warmer tone."
    responses = iter([
        _reply({"1": "ANNA"}, [{"id": "ANNA", "name": "Anna", "voice_hint": "clear"}]) | {
            "character_voices": {"ANNA": {"voice_direction": direction, "gender": "female"}},
            "delivery": {"1": {"direction": "with a trace of doubt"}},
        },
        _reply({"1": "ANN"}, [{"id": "ANN", "name": "Ann", "aliases": ["Anna"]}]) | {
            "character_voices": {"ANN": {"voice_direction": "Slightly deeper tone.", "gender": "male"}},
            "delivery": {"1": {"direction": "quietly"}},
        },
    ])
    prompts = []

    def post(_http, _url, **kwargs):
        prompts.append(kwargs["json"]["messages"])
        return _response(next(responses))

    monkeypatch.setattr(requests.Session, "post", post)
    result = web_backend.analyze_book_dialogue(session, "test-key", performance_cues=True, character_voices=True)
    assert len(prompts) == 2
    assert 'cue="true"' in prompts[0][1]["content"]
    system_prompt = prompts[0][0]["content"]
    assert all(f"- {choice}" in system_prompt for choice in CHARACTER_VOICE_DIRECTIONS)
    assert 'Use "Neutral delivery." for most characters.' in system_prompt
    assert "Situational\nperformance belongs to the individual quote." in system_prompt
    assert 'Use "unknown" if the evidence is ambiguous' in prompts[0][0]["content"]
    assert "Characters needing a voice direction:" not in prompts[1][1]["content"]
    anna = result.roster.get("ANNA")
    assert result.roster.ids == {"ANNA"}
    assert anna.aliases == ("Ann",)
    assert anna.voice_hint == "clear"
    assert anna.voice_direction == direction
    assert anna.gender == "female"
    assert result.attributions.by_first_line(book)[1].delivery == DeliveryCue("with a trace of doubt")
    assert result.attributions.by_first_line(book)[3].delivery == DeliveryCue("quietly")


def test_character_voice_backfill_refreshes_legacy_direction_without_reanalyzing_quotes(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION"), _section("SECOND-SECTION"), _section("THIRD-SECTION")])
    existing = AttributionsFile(tuple(
        RangeAttribution(start, end, UtteranceUtil.range_text_hash(book, start, end), (
            UtteranceAttribution((end - 1,), speaker, uncertain=True, delivery=DeliveryCue("quietly")),
        ))
        for (start, end), speaker in zip(book.section_ranges(), ("ANNA", "BETH", "BETH"), strict=True)
    ))
    anna = RosterCharacter("ANNA", "Anna", voice_direction=NEUTRAL_CHARACTER_VOICE_DIRECTION)
    beth = RosterCharacter(
        "BETH", "Beth", aliases=("Elizabeth",), merged_ids=("ELIZABETH",),
        voice_direction="A slight New Zealand accent and a florid upbeat cadence.",
    )
    session = web_backend.BookSession(book, str(tmp_path), roster=Roster((anna, beth)), attributions=existing)
    direction = "Slightly more measured pace."
    prompts = []

    def post(_http, _url, **kwargs):
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return _response(_reply({"1": "ANNA"}, [{"id": "ANNA", "name": "Somebody else"}]) | {
            "character_voices": {
                "ANNA": {"voice_direction": "Slightly brighter tone."},
                "ELIZABETH": {"voice_direction": direction, "gender": "female"},
            },
            "delivery": {"1": {"after": "sighs"}},
        })

    monkeypatch.setattr(requests.Session, "post", post)
    unchanged = web_backend.analyze_book_dialogue(session, "", performance_cues=True)
    assert not prompts
    assert unchanged.roster == session.roster
    result = web_backend.analyze_book_dialogue(session, "test-key", performance_cues=True, character_voices=True)
    assert len(prompts) == 1
    assert 'Characters needing a voice direction: ["BETH"]' in prompts[0]
    assert "SECOND-SECTION" in prompts[0] and "THIRD-SECTION" not in prompts[0]
    assert '<q speaker="BETH">' in prompts[0]
    assert '<q id=' not in prompts[0]
    assert result.attributions == existing
    assert result.roster.get("ANNA") == anna
    assert result.roster.get("BETH") == replace(beth, voice_direction=direction, gender="female")
    row = next(row for row in web_backend.character_rows(result) if row.speaker_id == "BETH")
    assert row.voice_direction == direction
    assert row.gender == "female"
    assert "3 of 3" in result.attribution_notice

    # Using the original session exercises reading the successful profile from disk.
    cached = web_backend.analyze_book_dialogue(session, "", performance_cues=True, character_voices=True)
    assert len(prompts) == 1
    assert cached.roster == result.roster
    assert cached.attributions == existing


def test_missing_character_voice_is_retryable_without_reviewing_the_quote_again(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[_section("FIRST-SECTION")])
    session = web_backend.BookSession(book, str(tmp_path))
    responses = iter([
        _reply({"1": "SAM"}, [{"id": "SAM", "name": "Sam"}]) | {
            "character_voices": {"SAM": {"voice_direction": "Speak gruffly with a gravelly accent and a buoyant cadence."}},
            "delivery": {"1": {"after": "sighs"}},
        },
        {"character_voices": {"SAM": {"voice_direction": NEUTRAL_CHARACTER_VOICE_DIRECTION, "gender": 5}}},
    ])
    prompts = []

    def post(_http, _url, **kwargs):
        prompts.append(kwargs["json"]["messages"][1]["content"])
        return _response(next(responses))

    monkeypatch.setattr(requests.Session, "post", post)
    progress = []
    partial = web_backend.analyze_book_dialogue(
        session, "test-key", performance_cues=True, character_voices=True,
        on_progress=lambda done, total, detail: progress.append((done, total, detail)),
    )
    assert all((done, total) == (0, 1) for done, total, _ in progress)
    assert "missing 1 character voice direction" in progress[-1][2]
    assert partial.roster.get("SAM").voice_direction == ""
    assert partial.attributions.by_first_line(book)[1].delivery == DeliveryCue(after="sighs")
    assert "0 of 1" in partial.attribution_notice
    assert "Missing 1 character voice direction" in partial.attribution_notice

    opt_out = web_backend.analyze_book_dialogue(partial, "", performance_cues=True)
    assert "1 of 1" in opt_out.attribution_notice
    assert len(prompts) == 1

    complete = web_backend.analyze_book_dialogue(partial, "test-key", performance_cues=True, character_voices=True)
    assert len(prompts) == 2
    assert '<q id=' not in prompts[-1]
    assert complete.attributions == partial.attributions
    assert complete.roster.get("SAM").voice_direction == NEUTRAL_CHARACTER_VOICE_DIRECTION
    assert complete.roster.get("SAM").gender == "unknown"
    assert "1 of 1" in complete.attribution_notice


def test_partial_saved_quote_does_not_move_deliveries_or_override_existing_line(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    book = Book(sections=[BookSection([
        _group("“First. ", Reason.SENTENCE, dialogue=True),
        _group("Second.” ", Reason.PARAGRAPH, dialogue=True),
        _group("“Third.” ", Reason.PARAGRAPH, dialogue=True),
    ])])
    fragment = UtteranceAttribution((0,), "ANNA", delivery=DeliveryCue(before="sighs"))
    existing = AttributionsFile((RangeAttribution(0, 1, UtteranceUtil.range_text_hash(book, 0, 1), (fragment,)),))
    session = web_backend.BookSession(
        book, str(tmp_path), roster=Roster((RosterCharacter("ANNA", "Anna"),)), attributions=existing,
    )

    def post(_http, _url, **kwargs):
        passage = kwargs["json"]["messages"][1]["content"]
        assert '<q id="1">“First. Second.”</q>' in passage
        assert '<q id="2" cue="true">“Third.”</q>' in passage
        return _response(_reply({"1": "UNKNOWN", "2": "ANNA"}) | {"delivery": {
            "1": {"after": "gasps"}, "2": {"after": "chuckles"},
        }})

    monkeypatch.setattr(requests.Session, "post", post)
    result = web_backend.analyze_book_dialogue(session, "test-key", performance_cues=True)
    quotes = result.attributions.by_first_line(book)
    assert quotes[0] == fragment
    assert quotes[1].speaker == "UNKNOWN"
    assert quotes[1].delivery is None
    assert quotes[2].delivery == DeliveryCue(after="chuckles")


def test_invalid_optional_delivery_does_not_reject_valid_labels():
    data = ('{"characters":[],"speakers":{"1":"UNKNOWN","2":"UNKNOWN","3":"NOT_SPEECH"},'
            '"delivery":{"1":{"after":"laughs","after":"coughs"},'
            '"2":{"direction":"(injected tag)","before":" GIGGLES ","after":42},'
            '"3":{"after":"sighs"},"4":{"after":"gasps"}}}')
    _, parsed = parse_chunk_reply(data, Roster(()), {1, 2, 3}, cue_requests={1, 2, 3})
    assert parsed.speakers == {1: "UNKNOWN", 2: "UNKNOWN", 3: "NOT_SPEECH"}
    assert parsed.delivery == {1: DeliveryCue(), 2: DeliveryCue(before="giggles")}
    assert not DeliveryCue(direction="x" * 81, before="sings")


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


def test_local_qwen_downloads_tags_and_unloads_then_reuses_saved_analysis(tmp_path, monkeypatch):
    from tts_audiobook_tool.web_local_tagger import CONTEXT_LENGTH, LOCAL_QWEN_MODEL, local_qwen

    monkeypatch.setattr(L, "logger", logging.getLogger(__name__), raising=False)
    calls = []
    reply = _reply({"1": "ANNA"}, [{"id": "ANNA", "name": "Anna"}])
    reply["delivery"] = {"1": {"direction": "with relief"}}

    def post(http, url, **kwargs):
        payload = kwargs["json"]
        assert not http.trust_env and not kwargs["allow_redirects"]
        assert url.startswith("http://127.0.0.1:11434/api/")
        assert payload["model"] == LOCAL_QWEN_MODEL
        route = url.split("/")[-1]
        calls.append(route)
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        response._content = b"{}"
        if route == "show":
            response.status_code = 404
        elif route == "pull":
            response._content = b'{"status":"success"}\n'
        elif route == "chat":
            assert payload["messages"][0]["content"] == PERFORMANCE_CUE_SYSTEM_PROMPT
            assert payload["options"]["num_ctx"] == CONTEXT_LENGTH
            assert payload["truncate"] is False and payload["format"] == "json"
            response._content = json.dumps({
                "done": True, "message": {"content": json.dumps(reply)}, "done_reason": "stop",
            }).encode()
        elif route == "generate":
            assert payload["keep_alive"] == 0
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    book = Book(sections=[_section("LOCAL")])
    session = web_backend.BookSession(book, str(tmp_path))
    for _ in range(2):
        with local_qwen(lambda _line: None) as tagger:
            session = web_backend.analyze_book_dialogue(session, "", local_tagger=tagger, performance_cues=True)
    assert calls == ["show", "pull", "chat", "generate"]
    saved = session.attributions.by_first_line(book)[1]
    assert saved.speaker == "ANNA" and saved.delivery == DeliveryCue("with relief")
