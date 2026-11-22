"""Gradio controls for the active narrator and its saved voice library."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from tts_audiobook_tool.app_types import RoomReverb
from tts_audiobook_tool.app_types.voice import VOLUME_DEFAULT, VOLUME_MAX, VOLUME_MIN
from tts_audiobook_tool.constants_config import IMPORT_LIMIT_SILENCE_DURATION_MAX, IMPORT_LIMIT_SILENCE_DURATION_MIN
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.web_model_options import voice_presets_for
from tts_audiobook_tool.web_models import ModelRunError, preview_model_reference, transcribe_model_reference
from tts_audiobook_tool.web_render import RenderVoice, WebRenderError
from tts_audiobook_tool.web_voices import (
    WebVoiceError, create_voice, delete_voice, list_voices, load_voice, move_voice, save_voice,
)


def voice_status(message: str | None = None) -> dict:
    return gr.update(value=message, visible=bool(message))


def bind_transcription(
    reference, language, transcript, status, retry, *, event_options, trim_inputs=(), draft=None, reset=(),
):
    def transcribe(path, code, start=0.0, end=None):
        if path is None:
            raise ModelRunError("Upload or record a reference clip first.")
        return transcribe_model_reference(path, language_code=code or "en", trim_start=start or 0.0, trim_end=end)

    def result(path, code, start=0.0, end=None):
        try:
            text = transcribe(path, code, start, end)
        except ModelRunError as exc:
            message = str(exc)
        except Exception:
            message = "Could not transcribe the reference audio."
        else:
            if text.strip():
                return gr.update(value=text, interactive=True), voice_status()
            message = "No speech was recognized. You can enter the transcript or retry."
        return gr.update(interactive=True), voice_status(message)

    def transcribe_new(path, code):
        yield (
            gr.update(value="", interactive=False), voice_status("Transcribing reference audio…"),
            *(value for _, value in reset),
        )
        yield (*result(path, code), *(gr.skip() for _ in reset))

    gr.on(
        [reference.upload, reference.stop_recording],
        transcribe_new, [reference, language], [transcript, status, *(field for field, _ in reset)],
        trigger_mode="multiple", show_progress="minimal", **event_options,
    )
    if draft is None:
        retry.click(result, [reference, language, *trim_inputs], [transcript, status], **event_options)
    else:
        retry.click(
            lambda path, code, voice: result(path, code, voice.trim_start, voice.trim_end),
            [reference, language, draft], [transcript, status], **event_options,
        )


class VoicePanel:
    def __init__(self):
        self.model_directions = {}
        self.model_presets = {}
        self.preset_choices = {}

        gr.Markdown("## Voice settings")
        gr.Markdown("Used for the narrator, text samples and voice chat.")
        self.selected = gr.Dropdown(label="Saved voice", choices=[], interactive=True)
        with gr.Row():
            self.move_up = gr.Button("Move up", size="sm", min_width=75, interactive=False)
            self.move_down = gr.Button("Move down", size="sm", min_width=85, interactive=False)
            self.remove = gr.Button("Remove voice", size="sm", min_width=100, interactive=False)
        self.name = gr.Textbox(label="Voice name")
        with gr.Row():
            self.add = gr.Button("Add to library", size="sm", min_width=100)
            self.save = gr.Button("Save voice", size="sm", min_width=80, interactive=False)
            self.new = gr.Button("New voice", size="sm", min_width=80)
        self.library_status = gr.Textbox(label="Voice library status", interactive=False, visible=False)
        self.reference = gr.Audio(
            sources=["upload", "microphone"], type="filepath", label="Reference audio", editable=False, buttons=[],
        )
        self.clear = gr.Button("Clear reference", size="sm")
        self.transcript = gr.Textbox(label="Reference transcript", lines=2, interactive=True)
        self.transcript_status = gr.Textbox(label="Transcription status", interactive=False, visible=False)
        with gr.Accordion("Transcription settings", open=False):
            self.language = gr.Textbox(value="en", label="Reference language code")
            self.retry = gr.Button("Retry transcription", size="sm")
        with gr.Accordion("Trim reference", open=False):
            with gr.Row():
                self.trim_start = gr.Number(value=0, minimum=0, step=0.01, label="Start (seconds)", min_width=130)
                self.trim_end = gr.Number(minimum=0, step=0.01, label="End (seconds)", min_width=130)
            gr.Markdown("Leave the end blank to use the rest of the clip. Retry transcription after adjusting the trim.")
            self.preview_trim = gr.Button("Preview trim", size="sm")
            self.preview = gr.Audio(label="Trimmed reference", interactive=False, visible=False, buttons=[])
            self.preview_status = gr.Textbox(label="Trim status", interactive=False, visible=False)
            self.max_reference_silence = gr.Number(
                label="Maximum internal silence (seconds)", minimum=IMPORT_LIMIT_SILENCE_DURATION_MIN,
                maximum=IMPORT_LIMIT_SILENCE_DURATION_MAX, step=0.05,
                info="Leave blank to keep existing pauses in the cloning audio.",
            )
        with gr.Accordion("Stereo and volume", open=False):
            self.pan = gr.Slider(-100, 100, value=0, step=1, label="Stereo position (left −100, right 100)")
            self.volume = gr.Slider(VOLUME_MIN, VOLUME_MAX, value=VOLUME_DEFAULT, step=1, label="Volume (%)")
            self.reverb = gr.Dropdown(
                choices=[(room.label, room.id) for room in RoomReverb], value=RoomReverb.OFF.id, label="Room reverb",
            )
        with gr.Accordion("Delivery instructions", open=False):
            for model in TtsModelType.get_local_items():
                spec = model.value
                if spec.can_instructions or spec.ui.get("instructions_message"):
                    self.model_directions[model.id] = gr.Textbox(label=spec.ui.get("proper_name") or model.id, lines=2)
            self.instructions = gr.Textbox(label="Default delivery instructions", lines=2)
        with gr.Accordion("Built-in voices", open=False):
            for model, label, empty_label in (
                ("qwen3tts_local", "Qwen3-TTS CustomVoice speaker", "Choose a speaker for CustomVoice"),
                ("pocket_local", "Pocket TTS predefined voice", "Use the reference audio"),
            ):
                choices = [
                    (empty_label, ""), *((preset.replace("_", " ").title(), preset) for preset in voice_presets_for(model)),
                ]
                self.preset_choices[model] = choices
                self.model_presets[model] = gr.Dropdown(label=label, choices=choices, value="")
        self.other_directions = gr.State(())
        self.other_presets = gr.State(())

        self.inputs = [
            self.reference, self.transcript, self.name, self.trim_start, self.trim_end,
            self.pan, self.volume, self.reverb, self.instructions, self.other_directions,
            *self.model_directions.values(),
            self.other_presets, *self.model_presets.values(), self.max_reference_silence,
        ]
        self.outputs = [*self.inputs, self.preview, self.preview_status, self.transcript_status]
        self.library_buttons = [self.save, self.remove, self.move_up, self.move_down]
        self.selection_outputs = [self.selected, *self.library_buttons]

    def snapshot(self, reference, transcript, name, start, end, pan, volume, reverb, instructions, other, *fields):
        direction_count = len(self.model_directions)
        directions, preset_fields = fields[:direction_count], fields[direction_count:]
        model_instructions = dict(other or ())
        model_instructions.update(
            (model_id, text.strip())
            for model_id, text in zip(self.model_directions, directions, strict=True) if text and text.strip()
        )
        presets = {}
        max_reference_silence = None
        if preset_fields:
            other_presets, *selected_presets, max_reference_silence = preset_fields
            presets.update(other_presets or ())
            presets.update((model, choice) for model, choice in zip(self.model_presets, selected_presets, strict=True) if choice)
        return RenderVoice(
            reference_path=reference, transcript=(transcript or "").strip(), instructions=(instructions or "").strip(),
            name=(name or "").strip(), trim_start=start or 0.0, trim_end=end, pan=pan / 100,
            volume=int(volume), room_reverb=reverb, model_instructions=tuple(model_instructions.items()),
            presets=tuple(presets.items()), max_reference_silence=max_reference_silence,
        )

    def show(self, voice: RenderVoice) -> tuple:
        directions = dict(voice.model_instructions)
        other_directions = tuple((model, text) for model, text in directions.items() if model not in self.model_directions)
        presets = dict(voice.presets)
        other_presets = tuple((model, name) for model, name in presets.items() if model not in self.model_presets)
        preset_updates = []
        for model, choices in self.preset_choices.items():
            value = presets.get(model, "")
            if value and all(choice != value for _, choice in choices):
                choices = [*choices, (f"{value} (saved)", value)]
            preset_updates.append(gr.update(choices=choices, value=value))
        return (
            self.reference.move_resource_to_block_cache(voice.reference_path),
            gr.update(value=voice.transcript, interactive=True),
            voice.name, voice.trim_start, voice.trim_end, round(100 * voice.pan), voice.volume, voice.room_reverb,
            voice.instructions, other_directions, *(directions.get(model, "") for model in self.model_directions),
            other_presets, *preset_updates, voice.max_reference_silence,
            gr.update(value=None, visible=False), voice_status(), voice_status(),
        )

    def reset_selection(self):
        return gr.update(value=None), *(gr.update(interactive=False) for _ in self.library_buttons)

    def refresh(self, selected=None, cast_selected=None) -> tuple:
        choices = [(item.name, item.voice_id) for item in list_voices()]
        valid_ids = {voice_id for _, voice_id in choices}
        selected = selected if selected in valid_ids else None
        selected_index = next((i for i, (_, voice_id) in enumerate(choices) if voice_id == selected), None)
        return (
            gr.update(choices=choices, value=selected),
            gr.update(interactive=selected is not None), gr.update(interactive=selected is not None),
            gr.update(interactive=selected_index is not None and selected_index > 0),
            gr.update(interactive=selected_index is not None and selected_index < len(choices) - 1),
            gr.update(choices=choices, value=cast_selected if cast_selected in valid_ids else None),
        )

    def bind(self, app, cast_selector, *, event_options):
        refresh_outputs = [*self.selection_outputs, cast_selector]
        library_outputs = [*refresh_outputs, self.library_status]
        selection_outputs = [*self.outputs, *library_outputs]

        def refresh_library():
            try:
                return *self.refresh(), voice_status()
            except WebVoiceError as exc:
                return *(gr.skip() for _ in refresh_outputs), voice_status(str(exc))

        def select_voice(selected, cast_selected):
            try:
                fields = self.show(load_voice(selected).voice if selected else RenderVoice())
                return *fields, *self.refresh(selected, cast_selected), voice_status()
            except (WebVoiceError, OSError) as exc:
                return (
                    *(gr.skip() for _ in self.outputs), *self.reset_selection(), gr.skip(),
                    voice_status(str(exc) if isinstance(exc, WebVoiceError) else "Could not open the saved reference."),
                )

        def adjust_library(operation, selected, cast_selected, *fields):
            try:
                if operation == "add":
                    summary = create_voice(self.snapshot(*fields), upload_root=self.reference.GRADIO_CACHE)
                    selected = summary.voice_id
                    message = f"Added {summary.name} to the voice library."
                elif operation == "save":
                    summary = save_voice(selected or "", self.snapshot(*fields), upload_root=self.reference.GRADIO_CACHE)
                    message = f"Saved {summary.name}."
                elif operation == "delete":
                    delete_voice(selected or "")
                    selected, message = None, "Removed the saved voice."
                else:
                    move_voice(selected or "", operation)
                    message = "Voice order updated."
                return *self.refresh(selected, cast_selected), voice_status(message)
            except (WebRenderError, WebVoiceError) as exc:
                return *(gr.skip() for _ in refresh_outputs), voice_status(str(exc))

        app.load(refresh_library, outputs=library_outputs, **event_options)
        selection_inputs = [self.selected, cast_selector]
        self.selected.input(select_voice, selection_inputs, selection_outputs, **event_options)
        self.new.click(lambda cast_id: select_voice(None, cast_id), [cast_selector], selection_outputs, **event_options)
        self.add.click(
            lambda role, *fields: adjust_library("add", None, role, *fields),
            [cast_selector, *self.inputs], library_outputs, **event_options,
        )
        self.save.click(
            lambda voice_id, role, *fields: adjust_library("save", voice_id, role, *fields),
            [*selection_inputs, *self.inputs], library_outputs, **event_options,
        )
        for button, operation in ((self.remove, "delete"), (self.move_up, -1), (self.move_down, 1)):
            button.click(
                lambda voice_id, role, op=operation: adjust_library(op, voice_id, role),
                selection_inputs, library_outputs, **event_options,
            )

        reset = (
            (self.trim_start, 0), (self.trim_end, None),
            (self.preview, gr.update(value=None, visible=False)), (self.preview_status, voice_status()),
        )
        bind_transcription(
            self.reference, self.language, self.transcript, self.transcript_status, self.retry,
            event_options=event_options, trim_inputs=(self.trim_start, self.trim_end), reset=reset,
        )
        gr.on(
            [self.reference.clear, self.clear.click],
            lambda: (None, gr.update(value="", interactive=True), voice_status(), *(value for _, value in reset)),
            outputs=[self.reference, self.transcript, self.transcript_status, *(field for field, _ in reset)],
            trigger_mode="multiple", **event_options,
        )

        def preview_trim(path, start, end):
            try:
                if path is None:
                    raise ModelRunError("Upload or record a reference clip first.")
                source = Path(preview_model_reference(path, trim_start=start or 0.0, trim_end=end))
                cached = None
                try:
                    cached = self.preview.move_resource_to_block_cache(source)
                finally:
                    if cached is None or Path(cached) != source:
                        source.unlink(missing_ok=True)
            except ModelRunError as exc:
                message = str(exc)
            except Exception:
                message = "Could not preview the reference audio."
            else:
                return gr.update(value=cached, visible=True), voice_status()
            return gr.update(value=None, visible=False), voice_status(message)

        self.preview_trim.click(
            preview_trim, [self.reference, self.trim_start, self.trim_end], [self.preview, self.preview_status],
            **event_options,
        )
        gr.on(
            [self.trim_start.input, self.trim_end.input],
            lambda: (gr.update(value=None, visible=False), voice_status()),
            outputs=[self.preview, self.preview_status], **event_options,
        )
