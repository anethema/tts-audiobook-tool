from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np
import soundfile
from ebooklib import epub

from tts_audiobook_tool import web_tools
from tts_audiobook_tool.app_types import ConcreteSegment, ConcreteWord
from tts_audiobook_tool.app_types.app_metadata import AppMetadata
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.constants import ABR_VERSION, APP_META_FLAC_FIELD, APP_META_MP4_MEAN, APP_META_MP4_TAG
from tts_audiobook_tool.l import L
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil


class TestWebTools(unittest.TestCase):
    def setUp(self):
        if getattr(L, "logger", None) is None:
            L.logger = logging.getLogger("test_web_tools")
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.uploads = self.root / "uploads"
        self.output = self.root / "result"
        self.uploads.mkdir()
        self.output.mkdir()

    def _write_audio(self, name, audio, rate=16000):
        path = self.uploads / name
        soundfile.write(path, audio, rate)
        return path

    def test_extracts_epub_text_to_download(self):
        source = self.uploads / "Livre 🌿.epub"
        ebook = epub.EpubBook()
        ebook.set_identifier("web-tools-book")
        ebook.set_title("The Book")
        ebook.set_language("en")
        chapter = epub.EpubHtml(title="Chapter One", file_name="chapter.xhtml", lang="en")
        chapter.content = "<h1>Chapter One</h1><p>The wind was quiet over the water.</p>"
        ebook.add_item(chapter)
        ebook.add_item(epub.EpubNav())
        ebook.spine = [chapter]
        ebook.toc = (chapter,)
        epub.write_epub(str(source), ebook)

        result = web_tools.epub_to_text(str(source), self.output, upload_root=self.uploads)
        self.assertEqual(result.path.name, "Livre 🌿.txt")
        self.assertIn("The wind was quiet over the water.", result.path.read_text(encoding="utf-8"))
        self.assertTrue(source.exists())

    def test_concatenates_in_natural_file_order_and_can_cancel(self):
        later = self._write_audio("Chapter 10.wav", np.full(1600, -0.5))
        earlier = self._write_audio("chapter 2.flac", np.full(1600, 0.5))
        progress = []
        with patch.object(web_tools, "monotonic", side_effect=[100, 100, 110]):
            result = web_tools.concatenate_audio(
                [str(later), str(earlier)], "flac", self.output, upload_root=self.uploads,
                on_counted_progress=lambda label, fraction: progress.append((label, fraction)),
            )
        self.assertEqual([fraction for _, fraction in progress], [0.0, 0.5])
        self.assertIn("ETA measuring", progress[0][0])
        self.assertIn("ETA 10s", progress[1][0])
        signal, rate = soundfile.read(result.path)
        self.assertEqual(rate, 16000)
        self.assertEqual(signal.size, 3200)
        np.testing.assert_allclose(signal[:1600], 0.5, atol=0.001)
        np.testing.assert_allclose(signal[1600:], -0.5, atol=0.001)

        cancel = Event()
        with self.assertRaisesRegex(web_tools.WebToolsError, "Stopped"):
            web_tools.concatenate_audio(
                [str(later), str(earlier)], "m4a", self.output,
                upload_root=self.uploads, cancel_event=cancel, on_progress=lambda _: cancel.set(),
            )
        self.assertFalse((self.output / "concatenated.m4a").exists())
        self.assertEqual(list(self.output.iterdir()), [result.path])

    def test_transcodes_flac_and_preserves_raw_metadata(self):
        source = self._write_audio("Book.flac", np.zeros(8000))
        metadata = AppMetadata(
            [TimedPhrase("Bonjour.", 0.0, 0.25)], "Titre 🌿", ABR_VERSION, [], "", False, {}, [], "generated",
        )
        raw = json.loads(metadata.to_json_string())
        raw["future_field"] = {"sentinel": 42}
        value = json.dumps(raw, ensure_ascii=False, indent=2)
        self.assertFalse(AudioMetaUtil.set_flac_custom_metadata_field(str(source), APP_META_FLAC_FIELD, value))
        result = web_tools.transcode_audiobook(str(source), self.output, upload_root=self.uploads)

        self.assertEqual(result.path.suffix, ".m4b")
        readback, error = AudioMetaUtil.get_mp4_metadata_tag(str(result.path), APP_META_MP4_MEAN, APP_META_MP4_TAG)
        self.assertFalse(error)
        self.assertEqual(readback, value)
        self.assertEqual(AppMetadata.load_from_file(str(result.path)).title, "Titre 🌿")

    def test_changes_reference_speed(self):
        source = self._write_audio("Speech.wav", np.sin(np.arange(48000) * 0.08) * 0.25)
        result = web_tools.change_reference_speed(str(source), 160, self.output, upload_root=self.uploads)
        self.assertEqual(result.path.name, "Speech_speed_160_percent.flac")
        self.assertAlmostEqual(soundfile.info(result.path).duration, 3 / 1.6, delta=0.15)
        with self.assertRaises(web_tools.WebToolsError):
            web_tools.change_reference_speed(str(source), 201, self.output, upload_root=self.uploads)

    def test_enhances_audiobook_with_console_alignment_and_cached_cpu_stt(self):
        audio = self._write_audio("Narration.flac", np.zeros(16000))
        text = self.uploads / "Source.txt"
        text.write_text("The small boat rolled gently.", encoding="utf-8")
        words = [ConcreteWord(index * 0.1, (index + 1) * 0.1, word, 1.0) for index, word in enumerate(
            ("The", "small", "boat", "rolled", "gently."),
        )]
        model_cache = self.root / "whisper-cache"
        model_cache.mkdir()
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (model_cache / name).touch()
        with patch("faster_whisper.utils.download_model", return_value=str(model_cache)) as local_model:
            with patch("tts_audiobook_tool.stt.FasterWhisperAdapter") as whisper:
                whisper.return_value.supported_languages = ["en"]
                whisper.return_value.transcribe.return_value = ([ConcreteSegment(0, 0.5, "", words)], None)
                progress = []
                result = web_tools.enhance_audiobook(
                    str(audio), str(text), "en", self.output, upload_root=self.uploads,
                    on_counted_progress=lambda label, fraction: progress.append((label, fraction)),
                )
        self.assertEqual(len(progress), 1)
        self.assertIn("ETA measuring", progress[0][0])
        self.assertEqual(progress[0][1], 0.0)
        self.assertTrue(local_model.call_args.kwargs["local_files_only"])
        self.assertEqual(whisper.call_args.kwargs["device"], "cpu")
        self.assertEqual(result.path.name, "Narration.abr.m4a")
        metadata = AppMetadata.load_from_file(str(result.path))
        self.assertEqual(metadata.type, "conversion")
        self.assertEqual(metadata.timed_phrases[0].text, "The small boat rolled gently.")
        self.assertGreater(metadata.timed_phrases[0].time_end, 0)

    def test_rejects_external_files_and_renamed_non_audio(self):
        outside = self.root / "private.epub"
        outside.touch()
        with self.assertRaises(web_tools.WebToolsError):
            web_tools.epub_to_text(str(outside), self.output, upload_root=self.uploads)
        invalid_audio = self.uploads / "Invalid.m4a"
        invalid_audio.write_text("Not an audio file", encoding="utf-8")
        with self.assertRaises(web_tools.WebToolsError):
            web_tools.concatenate_audio([str(invalid_audio)], "m4a", self.output, upload_root=self.uploads)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_panel_delivers_result_from_gradio_cache(self):
        try:
            import gradio as gr
            from gradio.state_holder import SessionState
        except ImportError:
            self.skipTest("Gradio is not installed in this environment.")
        from tts_audiobook_tool.web_tools_panel import build_tools_panel

        source = self.uploads / "Book.epub"
        source.touch()
        generated = []

        def convert(path, output_dir, *, upload_root):
            self.assertEqual(Path(path), source)
            self.assertEqual(Path(upload_root), self.uploads)
            result = output_dir / "Book.txt"
            result.write_text("A page of text.", encoding="utf-8")
            generated.append(result)
            return web_tools.ToolResult(result, "Text file ready.")

        async def exercise():
            with gr.Blocks(analytics_enabled=False) as app:
                build_tools_panel(event_options={"api_visibility": "private", "concurrency_id": "local-work"})
            app.has_launched = True
            action = next(fn for fn in app.fns.values() if any(
                app.blocks[block].value == "Extract text" for block, _ in fn.targets
            ))
            request = gr.Request(session_hash="test-tools-export")
            state = SessionState(app)
            payload = [{"path": str(source), "meta": {"_type": "gradio.FileData"}}]
            running = await app.process_api(action, payload, state, request=request)
            self.assertTrue(running["is_generating"])
            done = await app.process_api(action, payload, state, request=request, iterator=running["iterator"])
            await app.process_api(action, payload, state, request=request, iterator=done["iterator"])
            return done

        with patch.dict(os.environ, {"GRADIO_TEMP_DIR": str(self.uploads)}):
            with patch.object(web_tools, "epub_to_text", side_effect=convert):
                done = asyncio.run(exercise())
        cached = Path(done["data"][0]["value"]["path"])
        self.assertTrue(cached.is_relative_to(self.uploads))
        self.assertEqual(cached.read_text(encoding="utf-8"), "A page of text.")
        self.assertFalse(generated[0].exists())
