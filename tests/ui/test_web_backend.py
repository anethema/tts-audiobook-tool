from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ebooklib import epub
import requests

from tts_audiobook_tool import web_backend
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.l import L


KEY = "sk-test-request-only"
CHARACTERS_REPLY = {
    "characters": [
        {"id": "FIONA", "name": "Fiona", "is_narrator": True},
        {"id": "MILO", "name": "Milo", "voice_hint": "Quiet"},
    ],
}


def _response(*, status: int = 200, reply: dict | None = None, raw: dict | None = None) -> requests.Response:
    body = raw if raw is not None else {
        "choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}],
    }
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode("utf-8")
    return response


class TestWebBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if getattr(L, "logger", None) is None:
            L.logger = logging.getLogger("test_web_backend")

    def test_epub_import_preserves_logical_sections_and_separates_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fiction.epub"
            epub_book = epub.EpubBook()
            epub_book.set_identifier("sample-epub")
            epub_book.set_title("Fiona's Journey")
            epub_book.set_language("en")
            chapters = []
            for number in (1, 2):
                chapter = epub.EpubHtml(title=f"Chapter {number}", file_name=f"chapter{number}.xhtml", lang="en")
                chapter.content = "<h1>Chapter</h1><p>“Hello, Milo,” Fiona said.</p>"
                chapters.append(chapter)
                epub_book.add_item(chapter)
            epub_book.spine = chapters
            epub_book.toc = [epub.Link(chapter.file_name, chapter.title, chapter.file_name) for chapter in chapters]
            epub_book.add_item(epub.EpubNav())
            epub.write_epub(str(path), epub_book)

            session = web_backend.import_epub(str(path))
            self.addCleanup(web_backend.close_book_session, session)
            separate_session = web_backend.import_epub(str(path))
            self.addCleanup(web_backend.close_book_session, separate_session)

            self.assertEqual(session.book.title, "Fiona's Journey")
            self.assertEqual([section.title for section in session.book.sections], ["Chapter 1", "Chapter 2"])
            self.assertEqual(len(session.book.dialog_line_indices()), 2)
            self.assertTrue(session.book.segmentation_settings.dialog_segmentation)
            self.assertNotEqual(session.project_dir, separate_session.project_dir)
            web_backend.close_book_session(session)
            self.assertFalse(Path(session.project_dir).exists())
            self.assertTrue(Path(separate_session.project_dir).exists())

    def test_roster_and_attribution_send_key_only_in_official_https_header_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._book_session(directory, sections=2)
            responses = [
                _response(reply=CHARACTERS_REPLY),
                _response(reply={"speakers": {"1": "MILO"}}),
                _response(status=401, raw={"error": {"message": f"Incorrect API key: {KEY}"}}),
                _response(reply={"speakers": {"1": "OTHER"}}),
            ]
            # Stop any accidental real requests by patching the Requests boundary.
            with patch.object(requests.Session, "post", autospec=True, side_effect=responses) as post:
                with self.assertLogs(L.logger, logging.INFO) as logs:
                    identified = web_backend.identify_characters(session, KEY)
                    progress = []
                    partial = web_backend.attribute_dialog(
                        identified, KEY, on_progress=lambda done, total: progress.append((done, total)),
                    )
                    self.assertEqual(progress, [(0, 2), (1, 2)])
                    self.assertIn("1 of 2", partial.attribution_notice)
                    self.assertIn("API key", partial.attribution_notice)
                    complete = web_backend.attribute_dialog(partial, KEY, "custom-model", effort="")
                    self.assertEqual(post.call_count, 4)
                    for call in post.call_args_list:
                        http, url = call.args
                        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
                        self.assertFalse(http.trust_env)
                        self.assertFalse(call.kwargs["allow_redirects"])
                        self.assertEqual(call.kwargs["headers"], {"Authorization": f"Bearer {KEY}"})
                        payload = call.kwargs["json"]
                        self.assertNotIn("temperature", payload)
                        self.assertFalse(payload["store"])
                        self.assertNotIn(KEY, json.dumps(payload))
                    for call in post.call_args_list[:3]:
                        payload = call.kwargs["json"]
                        self.assertEqual(payload["model"], "gpt-6.1-terra")
                        self.assertEqual(payload["reasoning_effort"], "low")
                    override_payload = post.call_args.kwargs["json"]
                    self.assertEqual(override_payload["model"], "custom-model")
                    self.assertNotIn("reasoning_effort", override_payload)

            self.assertEqual(len(complete.attributions.ranges), 2)
            self.assertEqual([(row.speaker_id, row.line_count) for row in web_backend.character_rows(complete)],
                             [("FIONA", 3), ("MILO", 1)])
            self.assertNotIn(KEY, "\n".join(logs.output))
            self.assertNotIn(KEY, repr(complete))
            for artifact in Path(directory).rglob("*.json"):
                self.assertNotIn(KEY, artifact.read_text(encoding="utf-8"))

    def test_provider_errors_contain_no_credentials_and_do_not_follow_redirects(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._book_session(directory)
            scenarios = [
                (_response(status=400, raw={"error": {"code": "context_length_exceeded", "message": KEY}}),
                 "context limit"),
                (_response(status=307, raw={"error": {"message": KEY}}), "HTTP 307"),
                (_response(raw={"choices": [{"message": {"content": KEY}, "finish_reason": "stop"}]}),
                 "usable answer"),
                (requests.ConnectionError(KEY), "connect to OpenAI"),
            ]
            for response, expected_message in scenarios:
                with self.subTest(expected_message=expected_message):
                    with patch.object(requests.Session, "post", side_effect=[response]) as post:
                        with self.assertLogs(L.logger, logging.INFO) as logs:
                            with self.assertRaisesRegex(web_backend.WebBackendError, expected_message) as caught:
                                web_backend.identify_characters(session, KEY)
                    self.assertEqual(post.call_count, 1)
                    self.assertFalse(post.call_args.kwargs["allow_redirects"])
                    self.assertNotIn(KEY, str(caught.exception))
                    self.assertNotIn(KEY, "\n".join(logs.output))
                    self.assertFalse(list(Path(directory).rglob("*.json")))

    def test_missing_or_invalid_key_never_starts_a_request(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._book_session(directory)
            with patch.object(requests.Session, "post") as post:
                with patch.dict(os.environ, {"OPENAI_API_KEY": KEY}):
                    with self.assertRaisesRegex(web_backend.WebBackendError, "valid OpenAI API key"):
                        web_backend.identify_characters(session, "")
                    with self.assertRaisesRegex(web_backend.WebBackendError, "valid OpenAI API key"):
                        web_backend.identify_characters(session, f"{KEY}\n-injected")
                post.assert_not_called()

    @staticmethod
    def _book_session(directory: str, sections: int = 1) -> web_backend.BookSession:
        words = "A word " * 1300 if sections > 1 else "Fiona waved. "
        book = Book(sections=[
            BookSection([
                PhraseGroup([Phrase(words, Reason.PARAGRAPH)]),
                PhraseGroup([Phrase("“Come with me.” ", Reason.SECTION_BREAK)], is_dialog=True),
            ])
            for _ in range(sections)
        ])
        return web_backend.BookSession(book, directory)