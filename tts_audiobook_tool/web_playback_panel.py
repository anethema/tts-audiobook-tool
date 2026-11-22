"""Stream an audiobook line range to the browser."""
from __future__ import annotations

import asyncio
from contextlib import closing
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock

import gradio as gr
from gradio import processing_utils
from gradio.data_classes import FileData

from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.constants_config import PROJECT_MAX_RETRIES_DEFAULT
from tts_audiobook_tool.web_playback import WebPlaybackError, play_book, playback_range, wave_to_adts
from tts_audiobook_tool.web_remote_models import REMOTE_MODEL_ID, WebRemoteTtsError
from tts_audiobook_tool.web_render import WebRenderError
from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettingsError


_AUDIO_ELEMENT = "realtime-playback-audio"
_PAUSE_AUDIO = f"() => {{ document.querySelector('#{_AUDIO_ELEMENT} audio')?.pause(); return []; }}"


class PlaybackPanel:
    def __init__(self):
        with gr.Accordion("Realtime playback", open=False):
            with gr.Row():
                self.lines = gr.Textbox(label="Line range", value="all", info="Example: 3-12, or 3 to play from line 3.")
                self.save = gr.Checkbox(label="Save streamed audio", value=False)
            with gr.Row():
                self.start = gr.Button("Start playback")
                self.stop = gr.Button("Stop playback", interactive=False)
            self.audio = gr.Audio(
                label="Realtime audio", autoplay=True, streaming=True, interactive=False,
                buttons=[], elem_id=_AUDIO_ELEMENT,
            )
            self.download = gr.File(label="Download playback", interactive=False, visible=False)
            self.status = gr.Textbox(label="Playback status", interactive=False, visible=False)
        # Keep Gradio's player and HLS transport while using the same in-process
        # audio libraries as the console. Its default stream encoder needs a CLI.
        self.audio.stream_output = _stream_output

        async def combine_stream(stream, desired_output_format=None, only_file=False):
            name = "playback.aac"
            path = processing_utils.save_bytes_to_cache(b"".join(stream), name, cache_dir=self.audio.GRADIO_CACHE)
            return FileData(path=path, orig_name=name, is_stream=False)

        self.audio.combine_stream = combine_stream

    def bind(
            self, book, cast, model, validate_audio, batch_size, model_settings, strictness, max_retries, voice_panel, *,
            event_options, runtime_settings=None, remote_panel=None,
    ):
        active: dict[str | None, Event] = {}
        guard = Lock()
        runtime_inputs = () if runtime_settings is None else runtime_settings.inputs
        remote_inputs = () if remote_panel is None else remote_panel.selection_inputs
        runtime_end = len(runtime_inputs)
        remote_end = runtime_end + len(remote_inputs)

        def stop(request: gr.Request):
            with guard:
                if cancel := active.get(request.session_hash):
                    cancel.set()

        def start(session, voices, selected, validate, batch, settings, tolerance, retries, lines, save,
                  request: gr.Request, *fields):
            if session is None or not selected:
                yield None, gr.skip(), _status("Open a book and choose a speech model."), gr.skip(), gr.skip()
                return
            try:
                first, last = playback_range(lines, len(session.book.phrase_groups))
                runtime = runtime_settings.snapshot(*fields[:runtime_end]) if runtime_settings is not None else None
                remote = None
                if selected == REMOTE_MODEL_ID:
                    if remote_panel is None:
                        raise WebPlaybackError("Refresh and choose a remote model before starting playback.")
                    remote = remote_panel.selection_for(*fields[runtime_end:remote_end])
                narrator = voice_panel.snapshot(*fields[remote_end:])
            except (WebPlaybackError, WebRemoteTtsError, WebRenderError, WebRuntimeSettingsError) as error:
                yield None, gr.skip(), _status(str(error)), gr.skip(), gr.skip()
                return

            cancel = Event()
            with guard:
                active[request.session_hash] = cancel
            try:
                yield (
                    None, gr.update(value=None, visible=False), _status("Preparing playback…"),
                    gr.update(interactive=False), gr.update(interactive=True),
                )
                with TemporaryDirectory(prefix="audiobook-playback-download-") as directory:
                    file = Path(directory) / f"playback-lines-{first}-{last}.flac"
                    writer = None
                    download = None
                    complete = False
                    count = 0
                    try:
                        if save:
                            from tts_audiobook_tool.sound.audio_writers import open_writer

                            writer = open_writer(str(file), APP_SAMPLE_RATE, 2)
                        chunks = play_book(
                            selected, session, {**voices, "NARRATOR": narrator}, first=first, last=last,
                            validate_audio=validate, batch_size=batch or 1, model_options=(settings or {}).get(selected),
                            strictness=tolerance or None, max_retries=retries if retries is not None else PROJECT_MAX_RETRIES_DEFAULT,
                            cancel_event=cancel, runtime_settings=runtime, remote_selection=remote,
                        )
                        with closing(chunks):
                            for take in chunks:
                                if cancel.is_set():
                                    raise WebPlaybackError("Playback stopped.")
                                if writer is not None:
                                    import soundfile

                                    data, _ = soundfile.read(BytesIO(take.wave), dtype="float32", always_2d=True)
                                    writer.write(data)
                                count += 1
                                yield take.wave, gr.skip(), _status(f"Audio ready through line {take.line:,}."), gr.skip(), gr.skip()
                        complete = True
                        message = f"Audio ready for lines {first:,}–{last:,}."
                    except WebPlaybackError as error:
                        message = str(error)
                    except Exception:
                        message = "Could not finish playback."
                    finally:
                        if writer is not None:
                            try:
                                if count and (complete or cancel.is_set()):
                                    writer.close()
                                    download = self.download.move_resource_to_block_cache(str(file))
                            except Exception:
                                message = "Could not save streamed audio."
                            finally:
                                writer.abort()
                yield (
                    gr.skip(), gr.update(value=download, visible=download is not None), _status(message),
                    gr.update(interactive=True), gr.update(interactive=count > 0),
                )
            finally:
                with guard:
                    active.pop(request.session_hash, None)

        self.start.click(
            start,
            [book, cast, model, validate_audio, batch_size, model_settings, strictness, max_retries,
             self.lines, self.save, *runtime_inputs, *remote_inputs, *voice_panel.inputs],
            [self.audio, self.download, self.status, self.start, self.stop], **event_options,
        )
        self.stop.click(
            stop, api_visibility="private", queue=False, show_progress="hidden",
            js=_PAUSE_AUDIO,
        )


def _status(message: str):
    return gr.update(value=message, visible=True)


async def _stream_output(value, output_id, first_chunk):
    file = {"path": output_id, "is_stream": True, "orig_name": "playback.aac", "meta": {"_type": "gradio.FileData"}}
    if value is None:
        return None, file
    audio, duration = await asyncio.to_thread(wave_to_adts, value)
    return {"data": audio, "duration": duration, "extension": ".aac"}, file
