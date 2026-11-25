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

    def test_rendering_last_page_only_reads_text_for_that_page(self):
        class OffPageLine(PhraseGroup):
            @property
            def text(self):
                raise AssertionError("Rendering this page read a line on another page")

        off_page = OffPageLine([Phrase("Earlier line.", Reason.SENTENCE)])
        last = PhraseGroup([Phrase("Final line.", Reason.SECTION_BREAK)])
        session = BookSession(Book([BookSection([off_page] * 50_600 + [last])]), "large-project")
        self.assertEqual(web_line_voices.line_voice_rows(session, 0, 2025), [[50_601, "Automatic", "Final line."]])

    def test_browser_pages_a_large_section_and_applies_global_voice_ranges(self):
        try:
            import gradio as gr
            from gradio.state_holder import SessionState
        except ImportError:
            self.skipTest("Gradio is not installed in this environment.")
        from tts_audiobook_tool import web_line_voice_panel

        long_section = BookSection([
            PhraseGroup([Phrase(f"Line {number}.", Reason.SENTENCE)])
            for number in range(1, 50_605)
        ], "Long chapter")
        last_section = BookSection([PhraseGroup([Phrase("Closing line.", Reason.SECTION_BREAK)])], "Coda")
        session = BookSession(Book([long_section, last_section]), "large-project")

        async def exercise(voice_id):
            with gr.Blocks(analytics_enabled=False) as app:
                book = gr.State(None)
                panel = web_line_voice_panel.LineVoicePanel()
                panel.bind(app, book, event_options={"api_visibility": "private", "concurrency_id": "local-work"})
            app.has_launched = True
            state = SessionState(app)
            state[book._id] = session
            request = gr.Request(session_hash="test-line-voice")
            self.assertIsInstance(panel.lines, gr.Radio)
            props = {component["id"]: dict(component["props"]) for component in app.config["components"]}

            async def event(component, trigger):
                callbacks = [fn for fn in app.fns.values() if (component._id, trigger) in fn.targets]
                self.assertTrue(callbacks, f"No {trigger} callback on {component}")
                changed = set()
                for fn in callbacks:
                    result = await app.process_api(
                        fn, [None if item.stateful else props[item._id].get("value") for item in fn.inputs],
                        state, request=request,
                    )
                    changed.update(result["changed_state_ids"])
                    for item, update in zip(fn.outputs, result["data"], strict=True):
                        if item.stateful:
                            continue
                        if isinstance(update, dict) and update.get("__type__") == "update":
                            props[item._id].update(update)
                        else:
                            props[item._id]["value"] = update
                return changed

            def line_choices():
                return [tuple(choice) for choice in props[panel.lines._id]["choices"]]

            await event(book, "change")
            self.assertEqual(len(line_choices()), 0)
            await event(panel.panel, "expand")
            self.assertEqual([number for _, number in line_choices()], list(range(1, 26)))
            await event(panel.next_page, "click")
            self.assertEqual([number for _, number in line_choices()], list(range(26, 51)))
            await event(panel.previous_page, "click")
            self.assertEqual(line_choices()[0], ("1 · Automatic · Line 1.", 1))

            props[panel.page._id]["value"] = 2025
            await event(panel.page, "submit")
            self.assertEqual([number for _, number in line_choices()], [50_601, 50_602, 50_603, 50_604])
            self.assertEqual(line_choices()[-1], ("50,604 · Automatic · Line 50604.", 50_604))

            props[panel.lines._id]["value"] = line_choices()[1][1]
            await event(panel.lines, "input")
            self.assertEqual(props[panel.ranges._id]["value"], "50602")
            props[panel.ranges._id]["value"] = "25-26, 50602, 50605"
            props[panel.voice._id]["value"] = voice_id
            self.assertIn(book._id, await event(panel.assign, "click"))
            await event(book, "change")
            self.assertEqual(line_choices()[1], ("50,602 · Guest · Line 50602.", 50_602))
            assigned = state[book._id].book.phrase_groups
            self.assertTrue(all(assigned[line - 1].voice_index > 0 for line in (25, 26, 50_602, 50_605)))

            props[panel.ranges._id]["value"] = "50602"
            self.assertIn(book._id, await event(panel.clear, "click"))
            await event(book, "change")
            self.assertEqual(line_choices()[1], ("50,602 · Automatic · Line 50602.", 50_602))
            props[panel.section._id]["value"] = 1
            await event(panel.section, "input")
            self.assertEqual(line_choices(), [("50,605 · Guest · Closing line.", 50_605)])

            await event(panel.panel, "collapse")
            self.assertEqual(line_choices(), [])
            state[book._id] = _session()
            await event(book, "change")
            await event(panel.panel, "expand")
            self.assertEqual(line_choices()[0], ("1 · Automatic · A line of narration.", 1))

        with TemporaryDirectory() as directory:
            root = Path(directory) / "library"
            saved = web_voices.create_voice(RenderVoice(name="Guest"), root=root)
            with patch.object(web_line_voice_panel, "load_voice", partial(web_voices.load_voice, root=root)):
                with patch.object(web_line_voice_panel, "list_voices", partial(web_voices.list_voices, root=root)):
                    asyncio.run(exercise(saved.voice_id))
