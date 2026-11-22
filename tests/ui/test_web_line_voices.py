from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tts_audiobook_tool import web_line_voices, web_voices
from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.web_backend import BookSession, WebBackendError
from tts_audiobook_tool.web_render import RenderVoice


def _session():
    book = Book(
        sections=[
            BookSection([
                PhraseGroup([Phrase("A line of narration.", Reason.SENTENCE)]),
                PhraseGroup([Phrase("“Hello.”", Reason.SENTENCE)], is_dialog=True),
            ], "First chapter"),
            BookSection([], "An empty section"),
            BookSection([
                PhraseGroup([Phrase("The third line.", Reason.SENTENCE)], voice_index=1),
                PhraseGroup([Phrase("Last line.", Reason.SECTION_BREAK)]),
            ], "Last chapter"),
        ],
        title="Test Book", text_source_kind="epub", audio_source_kind="generated",
        segmentation_settings=BookSegmentationSettings(language_code="en", dialog_segmentation=True),
    )
    return BookSession(
        book, "example-project", warnings=("Stored warning",),
        roster=Roster((RosterCharacter("RUE", "Rue"),)),
        attributions=AttributionsFile((RangeAttribution(
            0, 4, UtteranceUtil.range_text_hash(book, 0, 4), (UtteranceAttribution((1,), "RUE"),),
        ),)),
        word_substitutions={"Herb": "erb"}, line_voices={1: RenderVoice(name="Earlier voice")},
    )


class TestLineVoices(unittest.TestCase):
    def test_assigns_inclusive_ranges_without_modifying_book_content_or_dialogue(self):
        session = _session()
        voice = RenderVoice(name="Guest", transcript="A sample.", pan=-0.4, model_instructions=(("qwen3tts_local", "Slowly"),))
        updated, count = web_line_voices.set_line_voice(session, "1, 3-4", voice)
        self.assertEqual(count, 3)
        self.assertEqual([group.voice_index for group in updated.book.phrase_groups], [2, 0, 2, 2])
        self.assertEqual([group.voice_index for group in session.book.phrase_groups], [0, 0, 1, 0])
        self.assertEqual(updated.line_voices, {1: session.line_voices[1], 2: voice})
        self.assertIs(updated.roster, session.roster)
        self.assertIs(updated.attributions, session.attributions)
        self.assertEqual(UtteranceUtil.range_text_hash(updated.book, 0, 4), UtteranceUtil.range_text_hash(session.book, 0, 4))
        self.assertEqual(updated.book.dialog_line_indices(), [1])
        self.assertEqual(updated.book.title, session.book.title)
        self.assertEqual(updated.book.audio_source_kind, session.book.audio_source_kind)
        self.assertEqual(updated.book.segmentation_settings, session.book.segmentation_settings)
        self.assertEqual([section.title for section in updated.book.sections], [section.title for section in session.book.sections])
        self.assertEqual(updated.book.phrase_groups[-1].phrases[0].reason, Reason.SECTION_BREAK)

    def test_reuses_snapshot_slots_and_clear_restores_automatic_voice(self):
        original = _session()
        selected = RenderVoice(name="Guest", transcript="First snapshot")
        session, _ = web_line_voices.set_line_voice(original, "1", selected)
        session, _ = web_line_voices.set_line_voice(session, "2", selected)
        self.assertEqual([line.voice_index for line in session.book.phrase_groups], [2, 2, 1, 0])
        self.assertEqual(web_line_voices.set_line_voice(session, "1-2", selected), (session, 0))

        changed = replace(selected, transcript="Second snapshot")
        session, _ = web_line_voices.set_line_voice(session, "4", changed)
        self.assertEqual(session.line_voices, {1: original.line_voices[1], 2: selected, 3: changed})
        cleared, count = web_line_voices.set_line_voice(session, "2-3", None)
        self.assertEqual(count, 2)
        self.assertEqual([line.voice_index for line in cleared.book.phrase_groups], [2, 0, 0, 3])
        self.assertEqual(cleared.line_voices, session.line_voices)
        self.assertEqual(cleared.attributions.by_first_line(cleared.book)[1].speaker, "RUE")

    def test_strict_ranges_never_partially_change_assignments(self):
        session = _session()
        voice = RenderVoice(name="Guest")
        for invalid in ("", "none", "3-1", "1, 4-99", "0, 2", "2, invalid", "²"):
            with self.subTest(invalid=invalid), self.assertRaises(WebBackendError):
                web_line_voices.set_line_voice(session, invalid, voice)
        self.assertEqual([group.voice_index for group in session.book.phrase_groups], [0, 0, 1, 0])
        updated, count = web_line_voices.set_line_voice(session, "all", voice)
        self.assertEqual(count, 4)
        self.assertEqual({group.voice_index for group in updated.book.phrase_groups}, {2})

    def test_lists_global_line_numbers_and_current_snapshots_by_section(self):
        session = _session()
        sections = web_line_voices.line_voice_sections(session)
        self.assertEqual([index for _, index in sections], [0, 2])
        self.assertEqual(web_line_voices.line_voice_rows(session, 2), [
            [3, "Earlier voice", "The third line."], [4, "Automatic", "Last line."],
        ])
        self.assertEqual(web_line_voices.line_voice_rows(session, 99), [])

    def test_browser_assign_and_clear_update_state_and_visible_rows(self):
        try:
            import gradio as gr
            from gradio.state_holder import SessionState
        except ImportError:
            self.skipTest("Gradio is not installed in this environment.")
        from tts_audiobook_tool import web_line_voice_panel

        async def exercise(voice_id):
            with gr.Blocks(analytics_enabled=False) as app:
                book = gr.State(None)
                panel = web_line_voice_panel.LineVoicePanel()
                panel.bind(app, book, event_options={"api_visibility": "private", "concurrency_id": "local-work"})
            app.has_launched = True
            state = SessionState(app)
            state[book._id] = _session()
            request = gr.Request(session_hash="test-line-voice")
            callbacks = {tuple(fn.targets): fn for fn in app.fns.values()}

            async def event(component, action, *values):
                fn = callbacks[((component._id, action),)]
                return await app.process_api(fn, list(values), state, request=request)

            load_choices = next(fn for fn in app.fns.values() if fn.name == "refresh_voices" and len(fn.targets) == 1)
            await app.process_api(load_choices, [], state, request=request)
            opened = await event(book, "change", None, None, None)
            self.assertTrue(opened["data"][0]["visible"])
            self.assertEqual(opened["data"][2]["data"][0], [1, "Automatic", "A line of narration."])
            assigned = await event(panel.assign, "click", None, 0, "2-3", voice_id)
            self.assertIn(book._id, assigned["changed_state_ids"])
            self.assertEqual(assigned["data"][1]["data"][1], [2, "Guest", "“Hello.”"])
            cleared = await event(panel.clear, "click", None, 0, "2")
            self.assertEqual(cleared["data"][1]["data"][1], [2, "Automatic", "“Hello.”"])
            self.assertEqual(state[book._id].book.phrase_groups[2].voice_index, 2)

        with TemporaryDirectory() as directory:
            root = Path(directory) / "library"
            saved = web_voices.create_voice(RenderVoice(name="Guest"), root=root)
            with patch.object(web_line_voice_panel, "load_voice", partial(web_voices.load_voice, root=root)):
                with patch.object(web_line_voice_panel, "list_voices", partial(web_voices.list_voices, root=root)):
                    asyncio.run(exercise(saved.voice_id))
