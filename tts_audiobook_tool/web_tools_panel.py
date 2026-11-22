"""Gradio controls for the browser's file tools."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock

import gradio as gr

from tts_audiobook_tool import web_tools


def build_tools_panel(*, event_options):
    active: dict[str | None, Event] = {}
    lock = Lock()

    def stop(request: gr.Request):
        with lock:
            if cancel := active.get(request.session_hash):
                cancel.set()

    def bind(operation, inputs, button_label, output_label, *, listen=False, stoppable=False):
        with gr.Row():
            button = gr.Button(button_label, variant="primary")
            cancel_button = gr.Button("Stop", interactive=False, visible=stoppable)
        status = gr.Textbox(label="Status", interactive=False, visible=False)
        output_type = gr.Audio if listen else gr.File
        output = output_type(label=output_label, interactive=False, visible=False)

        def run(request: gr.Request, progress=gr.Progress(), *values):
            cancel = Event()
            session = request.session_hash
            with lock:
                active[session] = cancel
            try:
                yield (
                    gr.update(value=None, visible=False), gr.update(value="Working…", visible=True),
                    gr.update(interactive=False), gr.update(interactive=stoppable),
                )
                try:
                    with TemporaryDirectory(prefix="tts-audiobook-tool-export-") as directory:
                        result = operation(
                            values, Path(directory), cancel, lambda description: progress(0, desc=description),
                        )
                        if cancel.is_set():
                            raise web_tools.WebToolsError("Stopped.")
                        path = output.move_resource_to_block_cache(str(result.path))
                except web_tools.WebToolsError as error:
                    path, message = None, str(error)
                except Exception:
                    path, message = None, "The file could not be processed."
                else:
                    message = result.message
            finally:
                with lock:
                    active.pop(session, None)
            yield (
                gr.update(value=path, visible=bool(path)), gr.update(value=message, visible=True),
                gr.update(interactive=True), gr.update(interactive=False),
            )

        button.click(run, inputs, [output, status, button, cancel_button], **event_options)
        if stoppable:
            cancel_button.click(stop, api_visibility="private", queue=False, show_progress="hidden")

    with gr.Accordion("Convert EPUB to text", open=True):
        epub = gr.File(label="EPUB", file_types=[".epub"])
        bind(
            lambda values, dest, _, __: web_tools.epub_to_text(*values, dest, upload_root=epub.GRADIO_CACHE),
            [epub], "Extract text", "Text file",
        )

    with gr.Accordion("Concatenate audio files", open=False):
        sources = gr.File(label="Audio files", file_types=list(web_tools.AUDIO_SUFFIXES), file_count="multiple")
        gr.Markdown("Files are combined in filename order.")
        format_choice = gr.Radio(choices=[("FLAC", "flac"), ("M4A", "m4a")], value="flac", label="Output format")
        bind(
            lambda values, dest, cancel, progress: web_tools.concatenate_audio(
                *values, dest, upload_root=sources.GRADIO_CACHE, cancel_event=cancel, on_progress=progress,
            ),
            [sources, format_choice], "Combine audio", "Combined audio file", stoppable=True,
        )

    with gr.Accordion("Transcode app FLAC to M4B", open=False):
        flac = gr.File(label="FLAC audiobook with synchronized text", file_types=[".flac"])
        bind(
            lambda values, dest, cancel, _: web_tools.transcode_audiobook(
                *values, dest, upload_root=flac.GRADIO_CACHE, cancel_event=cancel,
            ),
            [flac], "Transcode to M4B", "M4B audiobook", stoppable=True,
        )

    with gr.Accordion("Change reference clip speed", open=False):
        reference = gr.File(label="Reference clip", file_types=list(web_tools.AUDIO_SUFFIXES))
        speed = gr.Slider(50, 200, value=125, step=1, label="Speed (%)")
        bind(
            lambda values, dest, _, __: web_tools.change_reference_speed(
                *values, dest, upload_root=reference.GRADIO_CACHE,
            ),
            [reference, speed], "Change clip speed", "Adjusted clip", listen=True,
        )

    with gr.Accordion("Enhance an existing audiobook", open=False):
        gr.Markdown("Synchronize the source text with the audio using the cached Whisper model on the CPU.")
        audiobook = gr.File(label="Audiobook audio", file_types=list(web_tools.AUDIO_SUFFIXES))
        source_text = gr.File(label="Source text", file_types=[".epub", ".txt"])
        language = gr.Textbox(value="en", label="Language code")
        bind(
            lambda values, dest, cancel, progress: web_tools.enhance_audiobook(
                *values, dest, upload_root=audiobook.GRADIO_CACHE, cancel_event=cancel, on_progress=progress,
            ),
            [audiobook, source_text, language], "Synchronize audiobook", "Enhanced audiobook", stoppable=True,
        )
