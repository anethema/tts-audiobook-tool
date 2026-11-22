"""Browser controls for speech and validation runtime preferences."""
from __future__ import annotations

from dataclasses import dataclass
import platform

import gradio as gr

from tts_audiobook_tool.app_types import SttConfig, SttVariant
from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, AAC_BITRATES
from tts_audiobook_tool.web_runtime_settings import (
    WebRuntimeSettings, WebRuntimeSettingsError, latest_generation_log, load_runtime_settings, save_runtime_settings,
)


@dataclass(frozen=True)
class RuntimePanel:
    inputs: tuple[gr.Component, ...]

    @staticmethod
    def snapshot(
        force_cpu, whisper_variant, whisper_device, save_generation_logs, aac_bitrate=AAC_BITRATE_DEFAULT,
    ) -> WebRuntimeSettings:
        return WebRuntimeSettings(
            force_cpu=force_cpu, whisper_variant=whisper_variant or None, whisper_device=whisper_device or None,
            save_generation_logs=save_generation_logs, aac_bitrate=aac_bitrate,
        )


def create_runtime_panel(book: gr.State, *, private_event: dict, app: gr.Blocks | None = None) -> RuntimePanel:
    def read_preferences():
        try:
            return load_runtime_settings(), ""
        except WebRuntimeSettingsError as error:
            return WebRuntimeSettings(), str(error)

    defaults, initial_status = read_preferences()
    native_mlx = platform.system() == "Darwin" and platform.machine() == "arm64"
    with gr.Accordion("Speech runtime preferences", open=False):
        with gr.Row():
            force_cpu = gr.Checkbox(label="Force CPU for speech synthesis", value=defaults.force_cpu)
            save_logs = gr.Checkbox(label="Save audiobook generation logs", value=defaults.save_generation_logs)
        with gr.Row():
            whisper_model = gr.Dropdown(
                label="Whisper validation model", value=defaults.whisper_variant or "", choices=[
                    ("Automatic", ""),
                    *((f"{item.id} · {item.description}", item.id) for item in SttVariant if item is not SttVariant.DISABLED),
                ], info="Choose a model from the local cache, or let the app select its default.",
            )
            whisper_device = gr.Dropdown(
                label="Whisper validation device", value=defaults.whisper_device or "", choices=[
                    ("Automatic", ""), *((item.description, item.id) for item in SttConfig),
                ], visible=not native_mlx,
            )
        with gr.Row():
            bitrate = gr.Dropdown(label="Automatic M4B bitrate", choices=AAC_BITRATES, value=defaults.aac_bitrate)
            save = gr.Button("Save browser preferences")
        status = gr.Textbox(label="Browser preferences", value=initial_status, interactive=False, visible=bool(initial_status))
        with gr.Row():
            find_log = gr.Button("Find latest generation log")
            log = gr.File(label="Download generation log", interactive=False, visible=False)
        log_status = gr.Textbox(label="Generation log", interactive=False, visible=False)

    panel = RuntimePanel((force_cpu, whisper_model, whisper_device, save_logs, bitrate))

    def restore_preferences():
        settings, message = read_preferences()
        return (
            settings.force_cpu, settings.whisper_variant or "", settings.whisper_device or "",
            settings.save_generation_logs, settings.aac_bitrate, gr.update(value=message, visible=bool(message)),
        )

    def save_preferences(*values):
        try:
            save_runtime_settings(panel.snapshot(*values))
            message = "Saved browser preferences."
        except WebRuntimeSettingsError as error:
            message = str(error)
        return gr.update(value=message, visible=True)

    def show_log(session):
        try:
            path = latest_generation_log(session.project_dir) if session is not None else None
            if path:
                return (
                    gr.update(value=log.move_resource_to_block_cache(path), visible=True),
                    gr.update(value=None, visible=False),
                )
            message = "No saved generation log for this project."
        except OSError:
            message = "The latest generation log could not be opened."
        return gr.update(value=None, visible=False), gr.update(value=message, visible=True)

    find_log.click(show_log, [book], [log, log_status], **private_event)
    save.click(save_preferences, list(panel.inputs), status, **private_event)
    if app is not None:
        app.load(restore_preferences, outputs=[*panel.inputs, status], **private_event)
    return panel
