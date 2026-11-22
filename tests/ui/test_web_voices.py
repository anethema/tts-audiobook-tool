from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tts_audiobook_tool import web_voices
from tts_audiobook_tool.app_types import RoomReverb
from tts_audiobook_tool.web_render import RenderVoice


class TestWebVoices(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "voices"
        self.uploads = self.directory / "uploads"
        self.uploads.mkdir()

    def _clip(self, name, data):
        path = self.uploads / name
        path.write_bytes(data)
        return str(path)

    def _create(self, voice):
        return web_voices.create_voice(voice, root=self.root, upload_root=self.uploads)

    def _load(self, summary):
        return web_voices.load_voice(summary.voice_id, root=self.root).voice

    def test_ordered_profiles_round_trip_and_rename(self):
        self.assertEqual(web_voices.list_voices(root=self.root), [])
        voice = RenderVoice(
            self._clip("reference.wav", b"uploaded audio"), "Reference transcript.", "Read aloud.",
            name="  Narratrice 🌿  ", trim_start=0.25, trim_end=1.5, pan=-0.4, volume=80,
            room_reverb=RoomReverb.SMALL_DRR22.id,
            model_instructions=(("omnivoice", "Quiet"), ("qwen3", "Unhurried")),
            presets=(("qwen3tts_local", "serena"), ("pocket_local", "alba")),
            max_reference_silence=0.4,
        )
        first = self._create(voice)
        second = self._create(RenderVoice(name="Rue"))
        self.assertEqual(web_voices.list_voices(root=self.root), [first, second])
        Path(voice.reference_path).unlink()

        saved = self._load(first)
        self.assertEqual(saved, replace(voice, name="Narratrice 🌿", reference_path=saved.reference_path))
        self.assertEqual(Path(saved.reference_path).read_bytes(), b"uploaded audio")
        self.assertEqual(Path(saved.reference_path).parent, self.root / "clips")
        renamed = web_voices.save_voice(first.voice_id, replace(saved, name="Narrator"), root=self.root)
        self.assertEqual(renamed.voice_id, first.voice_id)
        self.assertEqual(self._load(renamed), replace(saved, name="Narrator"))

        web_voices.move_voice(second.voice_id, -1, root=self.root)
        web_voices.move_voice(second.voice_id, -1, root=self.root)
        self.assertEqual(web_voices.list_voices(root=self.root), [second, renamed])
        web_voices.move_voice(second.voice_id, 1, root=self.root)
        self.assertEqual(web_voices.list_voices(root=self.root), [renamed, second])

    def test_replacing_and_removing_profiles_keeps_cast_reference(self):
        original = self._clip("old.wav", b"old reference")
        first = self._create(RenderVoice(original, name="Narrator"))
        second = self._create(RenderVoice(original, name="Rue"))
        cast_voice = self._load(first)
        self.assertEqual(self._load(second).reference_path, cast_voice.reference_path)

        replacement = self._clip("new.wav", b"new reference")
        updated = web_voices.save_voice(
            first.voice_id, replace(cast_voice, reference_path=replacement),
            root=self.root, upload_root=self.uploads,
        )
        self.assertEqual(Path(self._load(updated).reference_path).read_bytes(), b"new reference")
        self.assertEqual(Path(self._load(second).reference_path).read_bytes(), b"old reference")

        web_voices.delete_voice(first.voice_id, root=self.root)
        web_voices.delete_voice(second.voice_id, root=self.root)
        self.assertEqual(web_voices.list_voices(root=self.root), [])
        self.assertEqual(Path(cast_voice.reference_path).read_bytes(), b"old reference")
        with self.assertRaises(web_voices.WebVoiceError):
            self._load(first)

    def test_publish_failure_preserves_previous_index(self):
        first = self._create(RenderVoice(name="Narrator"))
        original_replace = web_voices.os.replace

        def fail_index_replace(source, destination):
            if destination.name == "voices.json":
                raise OSError("simulated disk error")
            return original_replace(source, destination)

        with patch.object(web_voices.os, "replace", side_effect=fail_index_replace):
            with self.assertRaises(web_voices.WebVoiceError):
                web_voices.save_voice(
                    first.voice_id, RenderVoice(self._clip("voice.wav", b"new audio"), name="Renamed"),
                    root=self.root, upload_root=self.uploads,
                )
        self.assertEqual(web_voices.list_voices(root=self.root), [first])
        self.assertEqual(self._load(first), RenderVoice(name="Narrator"))
        self.assertEqual(list(self.root.glob(".voices-*.tmp")), [])

    def test_import_appends_collision_names_as_one_change(self):
        first = self._create(RenderVoice(name="Narrator"))
        console = RenderVoice(self._clip("console.wav", b"console audio"), name="narrator")
        outside = self.directory / "outside.wav"
        outside.write_bytes(b"outside audio")

        with self.assertRaises(web_voices.WebVoiceError):
            web_voices.import_voices(
                [console, RenderVoice(str(outside), name="Other")], root=self.root, upload_root=self.uploads,
            )
        self.assertEqual(web_voices.list_voices(root=self.root), [first])

        added = web_voices.import_voices(
            [console, replace(console, name="Narrator"), RenderVoice(name="Rue")],
            root=self.root, upload_root=self.uploads,
        )
        self.assertEqual([summary.name for summary in added], ["narrator (2)", "Narrator (3)", "Rue"])
        self.assertEqual(web_voices.list_voices(root=self.root), [first, *added])
        self.assertEqual(self._load(first), RenderVoice(name="Narrator"))
        self.assertEqual(self._load(added[0]).reference_path, self._load(added[1]).reference_path)

    def test_concurrent_creates_preserve_each_voice(self):
        profiles = [RenderVoice(name=f"Voice {number}") for number in range(8)]
        with ThreadPoolExecutor(max_workers=4) as workers:
            created = set(workers.map(self._create, profiles))
        self.assertEqual(set(web_voices.list_voices(root=self.root)), created)

    def test_rejects_untrusted_uploads_and_traversal(self):
        outside = self.directory / "outside.wav"
        outside.write_bytes(b"private file")
        with self.assertRaises(web_voices.WebVoiceError):
            self._create(RenderVoice(str(outside), name="Outside"))
        with self.assertRaises(web_voices.WebVoiceError):
            web_voices.load_voice("../outside.wav", root=self.root)
        with self.assertRaises(web_voices.WebVoiceError):
            self._create(RenderVoice(name=" "))

        index = self.root / "voices.json"
        index.write_text(json.dumps({"version": 1, "voices": [{
            "voice_id": "a" * 32, "voice": {"name": "Outside", "reference_path": "../outside.wav"},
        }]}), encoding="utf-8")
        with self.assertRaises(web_voices.WebVoiceError):
            web_voices.load_voice("a" * 32, root=self.root)

    def test_rejects_symlinks_in_uploads_and_library(self):
        outside = self.directory / "outside"
        outside.mkdir()
        private = outside / "private.wav"
        private.write_bytes(b"private file")
        alias = self.uploads / "alias"
        try:
            alias.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Could not create a test symlink: {exc}")
        with self.assertRaises(web_voices.WebVoiceError):
            self._create(RenderVoice(str(alias / private.name), name="Outside"))

        saved = self._create(RenderVoice(self._clip("voice.wav", b"audio"), name="Narrator"))
        reference = Path(self._load(saved).reference_path)
        reference.unlink()
        reference.symlink_to(private)
        with self.assertRaises(web_voices.WebVoiceError):
            self._load(saved)
