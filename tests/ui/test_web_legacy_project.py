from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
import unittest

from tts_audiobook_tool import web_legacy_project
from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings
from tts_audiobook_tool.app_types.book_serialization import book_to_project_text_json_dict
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import PROJECT_JSON_FILE_NAME, PROJECT_SPEC_VERSION, PROJECT_TEXT_EPUB_FILE_NAME, PROJECT_TEXT_FILE_NAME
from tts_audiobook_tool.l import L
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter, RosterFile
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.speaker_assign.voice_map import VoiceMapFile
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_backend import close_book_session


class TestWebLegacyProject(unittest.TestCase):
    def setUp(self):
        if getattr(L, "logger", None) is None:
            L.logger = logging.getLogger("test_web_legacy_project")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prefs = self.root / "prefs.json"
        self.project = self.root / "Console book"
        self.project.mkdir()
        self._write(self.prefs, {"project_dir": str(self.project)})
        self.book = Book(
            title="Book title",
            sections=[BookSection(
                title="Chapter",
                phrase_groups=[
                    PhraseGroup([Phrase("Milo shut the door. ", Reason.SENTENCE)]),
                    PhraseGroup(
                        [Phrase("“Not today.”", Reason.SECTION_BREAK)], voice_index=1, is_dialog=True,
                    ),
                ],
            )],
            segmentation_settings=BookSegmentationSettings(language_code="en", dialog_segmentation=True),
        )

    def test_read_only_import_preserves_book_cast_voices_and_model_choice(self):
        model = TtsModelType.require_by_id("chatterbox_local")
        self._write(self.project / PROJECT_JSON_FILE_NAME, {
            "version": PROJECT_SPEC_VERSION,
            "tts_model_type": model.id,
            "voice_references": [
                {"file_name": "narrator.wav", "transcript": "Full narrator text.", "crop_file_name": "trim.flac",
                 "crop_start": 1, "crop_end": 4, "crop_transcript": "Narrator crop."},
                {"file_name": "milo.flac", "transcript": "Milo transcript.",
                 "instructions": {model.id: "Quiet."}},
            ],
        })
        self._write(self.project / PROJECT_TEXT_FILE_NAME, book_to_project_text_json_dict(self.book))
        (self.project / "voice" / "crops").mkdir(parents=True)
        (self.project / "voice" / "narrator.wav").write_bytes(b"narrator")
        (self.project / "voice" / "crops" / "trim.flac").write_bytes(b"trim")
        (self.project / "milo.flac").write_bytes(b"legacy root voice")
        (self.project / PROJECT_TEXT_EPUB_FILE_NAME).write_bytes(b"project source")
        roster = Roster((RosterCharacter("MILO", "Milo"),))
        self.assertEqual(SpeakerAssignFileUtil.save_roster_file(str(self.project), RosterFile(roster)), "")
        valid = RangeAttribution(
            0, 2, UtteranceUtil.range_text_hash(self.book, 0, 2),
            (UtteranceAttribution((1,), "MILO"),),
            llm={"extra_params": {"private": "sample console credential"}},
        )
        stale = RangeAttribution(1, 2, "obsolete", (UtteranceAttribution((1,), "MILO"),))
        self.assertEqual(
            SpeakerAssignFileUtil.save_attributions_file(str(self.project), AttributionsFile((valid, stale))), "",
        )
        self.assertEqual(SpeakerAssignFileUtil.save_voices_file(str(self.project), VoiceMapFile({"MILO": 1})), "")
        original = self._contents()

        opened = self._load(model_choices=[(model.value.ui["proper_name"], "venv-local")])
        self.assertEqual(opened.name, self.project.name)
        self.assertEqual(book_to_project_text_json_dict(opened.session.book), book_to_project_text_json_dict(self.book))
        self.assertEqual(opened.model_id, "venv-local")
        self.assertEqual(opened.session.roster, roster)
        self.assertEqual(opened.session.attributions.speaker_of_each_line(opened.session.book), {1: "MILO"})
        self.assertEqual(len(opened.session.attributions.ranges), 1)
        self.assertEqual(opened.session.attributions.ranges[0].llm, {})
        self.assertFalse(Path(opened.session.project_dir).is_relative_to(self.project))
        self.assertEqual(opened.narrator.transcript, "Narrator crop.")
        self.assertEqual(Path(opened.narrator.reference_path).read_bytes(), b"trim")
        self.assertEqual(Path(opened.voices["MILO"].reference_path).read_bytes(), b"legacy root voice")
        self.assertEqual(opened.voices["MILO"].instructions, "Quiet.")
        self.assertEqual(Path(opened.source_epub).read_bytes(), b"project source")
        self.assertEqual(self._contents(), original)
        close_book_session(opened.session)
        self.assertFalse(Path(opened.session.project_dir).exists())
        self.assertEqual(self._contents(), original)

    def test_discovers_only_console_preferences_and_imports_legacy_flat_text(self):
        self._write(self.project / PROJECT_JSON_FILE_NAME, {
            "version": 2, "language_code": "es", "tts_model_type": "not-installed",
        })
        groups = [group.to_json_dict() for group in self.book.phrase_groups]
        self._write(self.project / PROJECT_TEXT_FILE_NAME, groups)
        recent = self.root / "Previous"
        recent.mkdir()
        self._write(recent / PROJECT_JSON_FILE_NAME, {"version": 1, "text": groups})
        unknown = self.root / "Different"
        unknown.mkdir()
        self._write(unknown / PROJECT_JSON_FILE_NAME, {"version": 1, "text": groups})
        self._write(self.prefs, {"project_dir": str(self.project), "last_project_dir": str(recent)})

        projects = web_legacy_project.list_console_projects(prefs_path=self.prefs)
        self.assertEqual([project.name for project in projects], [self.project.name, recent.name])
        self.assertTrue(all(str(self.root) not in project.project_id for project in projects))
        with self.assertRaisesRegex(web_legacy_project.WebLegacyProjectError, "Choose a project"):
            web_legacy_project.load_console_project(str(unknown), prefs_path=self.prefs)

        opened = self._load()
        self.assertEqual(opened.session.book.title, self.project.name)
        self.assertEqual(opened.session.book.segmentation_settings.language_code, "es")
        self.assertEqual(len(opened.session.book.phrase_groups), 2)
        self.assertEqual(opened.model_id, "")
        self.assertTrue(any("unavailable" in warning for warning in opened.session.warnings))

        previous = web_legacy_project.load_console_project(projects[1].project_id, prefs_path=self.prefs)
        self.addCleanup(close_book_session, previous.session)
        self.assertEqual(len(previous.session.book.phrase_groups), 2)

    def test_symlinked_reference_and_epub_and_path_traversal_are_skipped(self):
        outside = self.root / "external.wav"
        outside.write_bytes(b"keep outside the project")
        (self.project / "voice").mkdir()
        try:
            (self.project / "voice" / "narrator.wav").symlink_to(outside)
            (self.project / PROJECT_TEXT_EPUB_FILE_NAME).symlink_to(outside)
        except OSError:
            self.skipTest("creating symlinks is unavailable")
        self._write(self.project / PROJECT_JSON_FILE_NAME, {
            "voice_references": [{"file_name": "narrator.wav"}],
        })
        self._write(self.project / PROJECT_TEXT_FILE_NAME, book_to_project_text_json_dict(self.book))

        opened = self._load()
        self.assertIsNone(opened.narrator.reference_path)
        self.assertIsNone(opened.source_epub)
        self.assertEqual(outside.read_bytes(), b"keep outside the project")
        self._write(self.project / PROJECT_JSON_FILE_NAME, {
            "voice_references": [{"file_name": "../external.wav"}],
        })
        self.assertIsNone(self._load().narrator.reference_path)

    def test_rejects_future_project_format_and_project_text_symlink(self):
        settings = self.project / PROJECT_JSON_FILE_NAME
        self._write(settings, {"version": PROJECT_SPEC_VERSION + 1})
        with self.assertRaisesRegex(web_legacy_project.WebLegacyProjectError, "Update the app"):
            self._load()
        self._write(settings, {"version": PROJECT_SPEC_VERSION})
        outside = self.root / "external.json"
        self._write(outside, book_to_project_text_json_dict(self.book))
        try:
            (self.project / PROJECT_TEXT_FILE_NAME).symlink_to(outside)
        except OSError:
            self.skipTest("creating symlinks is unavailable")
        with self.assertRaisesRegex(web_legacy_project.WebLegacyProjectError, "no readable book text"):
            self._load()

    def _load(self, **kwargs):
        candidate = web_legacy_project.list_console_projects(prefs_path=self.prefs)[0]
        opened = web_legacy_project.load_console_project(candidate.project_id, prefs_path=self.prefs, **kwargs)
        self.addCleanup(close_book_session, opened.session)
        return opened

    def _contents(self):
        return {path.relative_to(self.project): path.read_bytes() for path in self.project.rglob("*") if path.is_file()}

    @staticmethod
    def _write(path: Path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")