import json
import re
from threading import Barrier, Event, Lock

import pytest

from tts_audiobook_tool import web_backend
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult


@pytest.fixture(autouse=True)
def short_chunks(tmp_path, monkeypatch):
    # Exercise the real planner with a small word budget so every section is a chunk.
    make = AssignPlan.make
    monkeypatch.setattr(AssignPlan, "make", staticmethod(lambda book, attrs: make(book, attrs, chunk_words=36)))
    monkeypatch.setattr(web_backend.recovery, "recovery_root", lambda: tmp_path / "recovery")


def _book(count):
    sections = []
    for n in range(count):
        text = f"“Chapter {n + 1}, " + "word " * 22 + "end.”"
        sections.append(BookSection([PhraseGroup([Phrase(text, Reason.SECTION_BREAK)], is_dialog=True)]))
    return Book(sections=sections)


def _request(prompt):
    before, passage = prompt.split("Current passage, starting at line ", 1)
    number, passage = passage.split(":\n", 1)
    passage = passage.split("End current passage.", 1)[0]
    cues = 'cue="true"' in passage
    speakers = bool(re.search(r'<q id="1"(?![^>]* speaker=)[^>]*>', passage))
    return int(number), speakers, cues, before


def _response(number, speakers, cues):
    payload = {
        "characters": [{"id": "ADA", "name": "Ada"}] if speakers and number == 1 else [],
        "speakers": {"1": "ADA"} if speakers else {},
        "uncertain": [],
    }
    if cues:
        payload["delivery"] = {"1": {"direction": f"with emphasis on line {number}"}}
    return LlmRequestResult(json.dumps(payload), "", 0.01)


def test_fresh_book_saves_out_of_order_cues_and_resumes_after_provider_failure(monkeypatch):
    book = _book(4)
    first = web_backend.new_book_session(book)
    checkpoint = web_backend._save_analysis_checkpoint
    another_cue_saved = Event()
    fail_first_cue = True
    calls = []
    active = peak = 0
    lock = Lock()

    def save(path, roster, attributions, **kwargs):
        checkpoint(path, roster, attributions, **kwargs)
        quotes = attributions.by_first_line(book)
        if len(quotes) == 4 and quotes[1].delivery is not None:
            another_cue_saved.set()

    def respond(prompt):
        number, speakers, cues, before = _request(prompt)
        calls.append((number, speakers, cues))
        if speakers and number > 1:
            assert '"id":"ADA"' in before and '<q speaker="ADA">' in before
        if cues:
            assert '<q id="1" speaker="ADA" cue="true">' in prompt
        if number == 1 and cues and fail_first_cue:
            # Force an out-of-order save while the first request remains in flight.
            assert another_cue_saved.wait(timeout=10)
            return LlmRequestResult("", "", 0.01, error=ConnectionError("offline"))
        return _response(number, speakers, cues)

    def send(_llm, prompt):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return respond(prompt)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(web_backend, "_send", send)
    monkeypatch.setattr(web_backend, "_save_analysis_checkpoint", save)
    partial = web_backend.analyze_book_dialogue(first, "test", performance_cues=True, request_concurrency=3)
    assert 2 <= peak <= 3
    assert [number for number, speakers, _ in calls if speakers] == [1, 2, 3, 4]
    assert all(speakers != cues for _, speakers, cues in calls)
    saved = partial.attributions.by_first_line(book)
    assert all(quote.speaker == "ADA" for quote in saved.values())
    assert saved[0].delivery is None and saved[1].delivery is not None
    outstanding = {line + 1 for line, quote in saved.items() if quote.delivery is None}
    web_backend.close_book_session(partial)

    calls.clear()
    fail_first_cue = False
    reopened = web_backend.new_book_session(book)
    try:
        complete = web_backend.analyze_book_dialogue(reopened, "test", performance_cues=True, request_concurrency=20)
        assert sorted(calls) == [(number, False, True) for number in sorted(outstanding)]
        assert complete.roster.ids == {"ADA"}
        assert [quote.delivery.direction for _, quote in sorted(complete.attributions.by_first_line(book).items())] == [
            f"with emphasis on line {number}" for number in range(1, 5)
        ]
    finally:
        web_backend.close_book_session(reopened)


def test_saved_speakers_allow_twenty_concurrent_cue_requests(tmp_path, monkeypatch):
    book = _book(24)
    session = web_backend.BookSession(book, str(tmp_path))
    active = peak = 0
    lock = Lock()
    first_wave = Barrier(20)
    cue_calls = []

    def send(_llm, prompt):
        nonlocal active, peak
        number, speakers, cues, _ = _request(prompt)
        if cues:
            with lock:
                active += 1
                peak = max(peak, active)
                cue_calls.append(number)
                wait_for_wave = len(cue_calls) <= 20
            try:
                if wait_for_wave:
                    first_wave.wait(timeout=10)
            finally:
                with lock:
                    active -= 1
        return _response(number, speakers, cues)

    monkeypatch.setattr(web_backend, "_send", send)
    labeled = web_backend.analyze_book_dialogue(session, "test")
    complete = web_backend.analyze_book_dialogue(labeled, "test", performance_cues=True, request_concurrency=20)
    assert peak == 20
    assert sorted(cue_calls) == list(range(1, 25))
    assert all(quote.delivery for quote in complete.attributions.by_first_line(book).values())


def test_parallel_resume_reuses_completed_chunks_from_old_combined_run(monkeypatch):
    book = _book(4)
    session = web_backend.new_book_session(book)
    calls = []
    fail_combined = True

    def send(_llm, prompt):
        number, speakers, cues, _ = _request(prompt)
        calls.append((number, speakers, cues))
        if number == 2 and fail_combined:
            return LlmRequestResult("", "", 0.01, error=ConnectionError("offline"))
        return _response(number, speakers, cues)

    monkeypatch.setattr(web_backend, "_send", send)
    partial = web_backend.analyze_book_dialogue(session, "test", performance_cues=True, request_concurrency=1)
    assert calls == [(1, True, True), (2, True, True)]
    web_backend.close_book_session(partial)

    calls.clear()
    fail_combined = False
    reopened = web_backend.new_book_session(book)
    try:
        complete = web_backend.analyze_book_dialogue(reopened, "test", performance_cues=True, request_concurrency=20)
        assert sorted(calls) == sorted((n, speakers, not speakers) for n in (2, 3, 4) for speakers in (True, False))
        assert len(complete.attributions.by_first_line(book)) == 4
        calls.clear()
        reused = web_backend.analyze_book_dialogue(complete, "", performance_cues=True, request_concurrency=20)
        assert not calls and reused.attributions == complete.attributions
    finally:
        web_backend.close_book_session(reopened)
