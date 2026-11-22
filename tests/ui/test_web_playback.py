from __future__ import annotations

import asyncio
from contextlib import nullcontext
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
import soundfile

from tts_audiobook_tool import web_playback, web_render
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery, RemoteTtsSnapshot
from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.constants import APP_SAMPLE_RATE, PROJECT_SOUND_SEGMENTS_SUBDIR
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.tts import Tts, WEB_REMOTE_ENTRY_ID_ENV
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.web_backend import BookSession
from tts_audiobook_tool.web_remote_models import REMOTE_MODEL_ID, RemoteTtsConfig, RemoteTtsSelection
from tts_audiobook_tool.web_render import RenderVoice


_RUN = "a" * 64


def _session(path):
    return BookSession(Book([BookSection([
        PhraseGroup([Phrase(f"Line {index}.", Reason.SENTENCE)]) for index in range(1, 4)
    ])]), str(path))


def _write_take(session, line, value, run_id=_RUN):
    index = line - 1
    digest = SoundSegmentUtil.calc_segment_hash(index, session.book.phrase_groups[index].text)
    folder = Path(session.project_dir) / "renders" / run_id / PROJECT_SOUND_SEGMENTS_SUBDIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"[{line:05d}] [{digest}] [breeze2] [voice] [0] take.flac"
    soundfile.write(path, np.full(4800, value, dtype=np.float32), APP_SAMPLE_RATE)
    return path


def _wave():
    data = BytesIO()
    soundfile.write(data, np.full((4800, 2), 0.125, dtype=np.float32), APP_SAMPLE_RATE, format="WAV")
    return data.getvalue()


class TestWebPlayback(unittest.TestCase):
    def test_native_line_range_semantics(self):
        self.assertEqual(web_playback.playback_range("all", 12), (1, 12))
        self.assertEqual(web_playback.playback_range("3", 12), (3, 12))
        self.assertEqual(web_playback.playback_range("3-7", 12), (3, 7))
        with self.assertRaises(web_playback.WebPlaybackError):
            web_playback.playback_range("8-2", 12)

    def test_reuses_matching_takes_and_generates_missing_only_in_staging(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = _session(root / "project")
            output = root / "output"
            output.mkdir()
            first = _write_take(project, 1, 0.125)
            third = _write_take(project, 3, -0.125)
            payload = {
                "first": 1, "last": 3, "validate_audio": False, "batch_size": 5, "strictness": None,
                "max_retries": 2, "model_options": None, "runtime_settings": None,
            }
            calls = []

            def generate(session, voices, **settings):
                calls.append(settings)
                self.assertTrue(web_playback._ready_path(output, 1).exists())
                stage = Path(session.project_dir) / "renders" / _RUN / PROJECT_SOUND_SEGMENTS_SUBDIR
                self.assertEqual((stage / third.name).read_bytes(), third.read_bytes())
                self.assertEqual(Path(session.project_dir).parent, output)
                _write_take(session, 2, 0.25)
                settings["on_progress"](2, 2)

            model = SimpleNamespace(id="breeze_tts_2_local", can_batch=lambda: True)
            with (
                patch.object(web_render, "_render_job", side_effect=lambda _: nullcontext()),
                patch.object(web_render, "_detect_model", return_value=model),
                patch.object(web_render, "_checkpoint_id", return_value=_RUN),
                patch.object(web_render, "_validated_model_options", return_value={}),
                patch.object(web_render, "render_book", side_effect=generate),
            ):
                web_playback._run_playback(output, project, {"NARRATOR": RenderVoice(volume=50)}, payload, Event())
                payload["first"] = payload["last"] = 3
                web_playback._run_playback(output, project, {"NARRATOR": RenderVoice(volume=50)}, payload, Event())
            self.assertEqual(len(calls), 1)
            self.assertEqual((calls[0]["first_line"], calls[0]["last_line"]), (2, 3))
            self.assertFalse(calls[0]["auto_concat"])
            self.assertEqual(set(first.parent.iterdir()), {first, third})
            for index, value in enumerate((0.0625, 0.125, -0.0625), start=1):
                wave, rate = soundfile.read(web_playback._ready_path(output, index))
                self.assertEqual(rate, APP_SAMPLE_RATE)
                np.testing.assert_allclose(wave[:100], np.full((100, 2), value), atol=0.001)

    def test_cancellation_stops_process_and_removes_staging(self):
        with TemporaryDirectory() as directory:
            project = _session(directory)
            cancel = Event()
            temporary = []
            process = SimpleNamespace(poll=lambda: None)

            def spawn(argv, **_):
                work = Path(argv[-1])
                temporary.append(work)
                web_playback._ready_path(work, 1).write_bytes(_wave())
                return process

            with (
                patch.object(web_playback, "_python_for", return_value="/test/python"),
                patch.object(web_playback.subprocess, "Popen", side_effect=spawn),
                patch.object(web_playback, "_stop") as stop,
            ):
                chunks = web_playback.play_book("venv-test", project, {}, first=1, last=3, cancel_event=cancel)
                self.assertEqual(next(chunks).line, 1)
                cancel.set()
                with self.assertRaisesRegex(web_playback.WebPlaybackError, "stopped"):
                    next(chunks)
            stop.assert_called_once_with(process)
            self.assertFalse(temporary[0].exists())

    def test_remote_playback_reuses_only_the_exact_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _session(root / "project")
            model = TtsModelType.require_by_id("higgs_v3_sglomni")
            selected = RemoteTtsSelection("https://first.example", model.id, "org/higgs")
            other = RemoteTtsSelection("https://other.example", model.id, "org/higgs")
            book, references = web_render._cast_book(session, {})
            checkpoints = [
                web_render._checkpoint_id(book, references, model.id, session.language_code, False, 1,
                                          remote_selection=selection)
                for selection in (selected, other)
            ]
            self.assertNotEqual(*checkpoints)
            _write_take(session, 1, 0.125, checkpoints[0])
            _write_take(session, 2, -0.125, checkpoints[1])
            executable = root / "venv-remote" / "bin" / "python"
            launched = []

            def discover(*, force):
                self.assertTrue(force)
                self.assertEqual(RemoteTtsDiscovery.get_base_url(), selected.endpoint)
                return RemoteTtsSnapshot(TtsBackendKind.SGL_OMNI, candidates=((model, selected.server_model_id),))

            def generate(staged, voices, **settings):
                self.assertEqual(settings["remote_selection"], selected)
                self.assertEqual(os.environ[WEB_REMOTE_ENTRY_ID_ENV], selected.server_model_id)
                self.assertEqual((settings["first_line"], settings["last_line"]), (2, 2))
                self.assertTrue(web_playback._ready_path(Path(staged.project_dir).parent, 1).is_file())
                _write_take(staged, 2, 0.25, checkpoints[0])
                settings["on_progress"](1, 1)

            def spawn(command, **_):
                launched.append(command[0])
                work = Path(command[-1])
                selection = json.loads((work / "input.json").read_text(encoding="utf-8"))["remote_selection"]
                self.assertEqual(RemoteTtsSelection(**selection), selected)
                web_playback._worker(work)
                return SimpleNamespace(poll=lambda: 0)

            with (
                patch.object(web_playback, "remote_client_python", return_value=executable),
                patch.object(web_playback, "_python_for", side_effect=AssertionError("Local worker requested")),
                patch.object(web_playback.subprocess, "Popen", side_effect=spawn),
                patch.object(web_playback.signal, "signal"),
                patch.object(RemoteTtsDiscovery, "refresh", side_effect=discover),
                patch.object(Tts, "init_local_model_type"), patch.object(Tts, "is_remote_mode", return_value=True),
                patch.object(web_render.ModelWorker, "shutdown"),
                patch.object(web_render, "_detect_model", side_effect=AssertionError("Local model requested")),
                patch.object(web_render, "_render_job", side_effect=lambda _: nullcontext()),
                patch.object(web_render, "render_book", side_effect=generate),
            ):
                chunks = list(web_playback.play_book(
                    REMOTE_MODEL_ID, session, {}, first=1, last=2, validate_audio=False, remote_selection=selected,
                ))
            self.assertEqual(launched, [str(executable)])
            self.assertEqual([chunk.line for chunk in chunks], [1, 2])
            for chunk, expected in zip(chunks, (0.125, 0.25), strict=True):
                data, _ = soundfile.read(BytesIO(chunk.wave), always_2d=True)
                np.testing.assert_allclose(data[:100], np.full((100, 2), expected), atol=0.001)

    def test_adts_uses_bundled_encoder_and_reports_decoded_duration(self):
        encoded, duration = web_playback.wave_to_adts(_wave())
        with av.open(BytesIO(encoded), mode="r", format="aac") as audio:
            frames = list(audio.decode(audio=0))
        self.assertAlmostEqual(duration, sum(frame.samples for frame in frames) / APP_SAMPLE_RATE)
        self.assertGreaterEqual(duration, 0.1)
        self.assertLess(duration, 0.15)

    def test_gradio_stream_and_download_without_starting_a_model(self):
        try:
            import gradio as gr
            from gradio.state_holder import SessionState
        except ImportError:
            self.skipTest("Gradio is not installed in this environment.")
        from tts_audiobook_tool import web_playback_panel, web_remote_panel

        async def exercise(directory, remote_selection):
            voice = SimpleNamespace(inputs=(), snapshot=lambda: RenderVoice())
            model_id = REMOTE_MODEL_ID if remote_selection is not None else "venv-test"
            with gr.Blocks(analytics_enabled=False) as app:
                book = gr.State(None)
                cast = gr.State({})
                model = gr.Dropdown(choices=[model_id])
                validate = gr.Checkbox()
                batch = gr.Number(precision=0)
                model_settings = gr.State({})
                strictness = gr.Dropdown(choices=[""])
                retries = gr.Number(precision=0)
                panel = web_playback_panel.PlaybackPanel()
                remote_panel = web_remote_panel.RemoteModelPanel() if remote_selection is not None else None
                panel.bind(book, cast, model, validate, batch, model_settings, strictness, retries, voice,
                           event_options={"api_visibility": "private", "concurrency_id": "local-work"},
                           remote_panel=remote_panel)
            app.has_launched = True
            state = SessionState(app)
            state[book._id] = _session(directory)
            callback = next(fn for fn in app.fns.values() if fn.targets == [(panel.start._id, "click")])
            values = [None, None, model_id, True, 1, None, "", 2, "1-2", True]
            if remote_panel is not None:
                state[remote_panel.state._id] = RemoteTtsConfig(
                    remote_selection.endpoint, remote_selection.backend_kind, (remote_selection,), remote_selection,
                )
                values.extend([remote_selection.endpoint, None])
            request = gr.Request(session_hash="playback-test")
            options = {"state": state, "request": request, "session_hash": request.session_hash}
            result = await app.process_api(callback, values, **options)
            self.assertTrue(result["data"][0]["is_stream"])
            self.assertIn("playlist.m3u8", result["data"][0]["path"])
            while result["is_generating"]:
                result = await app.process_api(callback, values, iterator=result["iterator"], **options)
            download = Path(result["data"][1]["value"]["path"])
            self.assertEqual(soundfile.info(download).frames, 9600)

        with TemporaryDirectory() as directory:
            with (
                patch.dict(os.environ, {"GRADIO_TEMP_DIR": directory}),
                patch.object(web_playback_panel, "play_book", side_effect=lambda *_, **__: (
                    web_playback.PlaybackChunk(line, _wave()) for line in (1, 2)
                )) as play,
            ):
                remote = RemoteTtsSelection("https://tts.example", "higgs_v3_sglomni", "org/higgs")
                for selection in (None, remote):
                    with self.subTest(remote=selection is not None):
                        asyncio.run(exercise(directory, selection))
                        self.assertEqual(play.call_args.kwargs["remote_selection"], selection)
