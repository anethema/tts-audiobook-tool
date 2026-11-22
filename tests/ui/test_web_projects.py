from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from tts_audiobook_tool import web_backend, web_projects
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.book_serialization import (
    book_to_project_text_json_dict,
    load_book_from_project_text_file,
)
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import PROJECT_TEXT_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.web_render import RenderVoice


class TestWebProjects(unittest.TestCase):
    def setUp(self):
        if getattr(L, "logger", None) is None:
            L.logger = logging.getLogger("test_web_projects")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "saved"
        self.book = Book(
            sections=[BookSection(
                [PhraseGroup([Phrase("“Hello, Rue.”", Reason.SECTION_BREAK)], is_dialog=True)],
                title="Chapter 1",
            )],
            title="A Book",
        )
        self.roster = Roster((RosterCharacter("RUE", "Rue"),))
        self.attributions = AttributionsFile((RangeAttribution(
            start=0, end=1,
            text_hash=UtteranceUtil.range_text_hash(self.book, 0, 1),
            utterances=(UtteranceAttribution((0,), "RUE"),),
        ),))
        self.session = web_backend.BookSession(
            self.book, str(self.directory), ("Import warning",), self.roster, self.attributions,
        )

    def test_saved_book_assets_and_analysis_round_trip(self):
        epub_path = self.directory / "uploaded.epub"
        audio_path = self.directory / "voice.wav"
        epub_path.write_bytes(b"EPUB upload")
        audio_path.write_bytes(b"audio upload")
        summary = web_projects.create_project(
            "  Adventures / at home\t ", self.session, narrator=RenderVoice(str(audio_path), "Hello."),
            voices={"RUE": RenderVoice(str(audio_path), "", "Quiet")}, model_id="chatterbox",
            source_epub=str(epub_path), validate_audio=False, batch_size=3, root=self.root,
        )
        self.assertEqual(summary.name, "Adventures at home")
        self.assertEqual(web_projects.list_projects(root=self.root), [summary])

        epub_path.unlink()
        audio_path.unlink()
        saved = self._open(summary.project_id)
        reopened = self._open(summary.project_id)
        self.assertEqual(saved.summary, summary)
        self.assertEqual(book_to_project_text_json_dict(saved.session.book), book_to_project_text_json_dict(self.book))
        self.assertEqual(saved.session.warnings, ("Import warning",))
        self.assertEqual(saved.session.roster, self.roster)
        self.assertEqual(saved.session.attributions, self.attributions)
        self.assertEqual(saved.model_id, "chatterbox")
        self.assertFalse(saved.validate_audio)
        self.assertEqual(saved.batch_size, 3)
        self.assertEqual(saved.voices["RUE"].instructions, "Quiet")
        self.assertEqual(saved.narrator.reference_path, saved.voices["RUE"].reference_path)
        self.assertEqual(Path(saved.narrator.reference_path).read_bytes(), b"audio upload")
        self.assertEqual(Path(saved.source_epub).read_bytes(), b"EPUB upload")
        self.assertEqual(saved.session.project_dir, reopened.session.project_dir)
        self.assertNotEqual(saved.session.attribution_dir, reopened.session.attribution_dir)

        project_text_path = self.root / summary.project_id / PROJECT_TEXT_FILE_NAME
        saved_book = load_book_from_project_text_file(str(project_text_path))
        self.assertIsInstance(saved_book, Book)
        self.assertEqual(book_to_project_text_json_dict(saved_book), book_to_project_text_json_dict(self.book))
        with patch.object(requests.Session, "post") as post:
            resumed = web_backend.attribute_dialog(saved.session, "sk-test")
            post.assert_not_called()
        self.assertEqual(resumed.attributions, self.attributions)
        web_backend.close_book_session(saved.session)
        self.assertTrue(Path(saved.session.project_dir).is_dir())
        self.assertTrue(Path(reopened.session.attribution_dir).is_dir())
        self.assertFalse(Path(saved.session.attribution_dir).exists())

    def test_failed_save_keeps_previous_snapshot_and_explicit_save_can_clear_assignments(self):
        summary = web_projects.create_project(
            "First version", self.session, narrator=RenderVoice(), voices={}, model_id="old", root=self.root,
        )
        changed = replace(self.session, attributions=AttributionsFile())

        with patch.object(web_projects.JsonSaveUtil, "save", return_value="simulated disk error"):
            with self.assertRaisesRegex(web_projects.WebProjectError, "Could not save"):
                web_projects.save_project(
                    summary.project_id, changed, narrator=RenderVoice(), voices={}, model_id="new", root=self.root,
                )
        saved = self._open(summary.project_id)
        self.assertEqual(saved.model_id, "old")
        self.assertEqual(saved.session.attributions, self.attributions)

        web_projects.save_project(
            summary.project_id, changed, narrator=RenderVoice(), voices={}, model_id="new", root=self.root,
        )
        reopened = self._open(summary.project_id)
        self.assertEqual(reopened.model_id, "new")
        self.assertEqual(reopened.session.attributions, AttributionsFile())

    def test_rejects_non_project_ids_and_symlinked_project_assets(self):
        audio_path = self.directory / "voice.wav"
        audio_path.write_bytes(b"only for upload")
        summary = web_projects.create_project(
            "Project", self.session, narrator=RenderVoice(str(audio_path)),
            voices={}, model_id="", root=self.root,
        )
        with self.assertRaisesRegex(web_projects.WebProjectError, "Choose a saved project"):
            web_projects.load_project("../outside", root=self.root)
        alias = self.root / ("a" * 32)
        try:
            alias.symlink_to(self.root / summary.project_id, target_is_directory=True)
        except OSError:
            self.skipTest("creating symlinks is unavailable")
        with self.assertRaisesRegex(web_projects.WebProjectError, "not found"):
            web_projects.load_project(alias.name, root=self.root)
        self.assertEqual(web_projects.list_projects(root=self.root), [summary])

        asset = self._open(summary.project_id).narrator.reference_path
        Path(asset).unlink()
        Path(asset).symlink_to(audio_path)
        with self.assertRaises(web_projects.WebProjectError):
            web_projects.load_project(summary.project_id, root=self.root)
        self.assertEqual(audio_path.read_bytes(), b"only for upload")

    def _open(self, project_id: str) -> web_projects.SavedProject:
        saved = web_projects.load_project(project_id, root=self.root)
        self.addCleanup(web_backend.close_book_session, saved.session)
        return saved