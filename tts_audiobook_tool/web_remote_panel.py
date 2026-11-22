"""Gradio controls for the remote speech client and its server models."""

from __future__ import annotations

from threading import Event, Lock

import gradio as gr

from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.web_model_setup import WebModelSetupError, install_remote_client, remote_client_install_supported
from tts_audiobook_tool.web_remote_models import (
    RemoteTtsConfig, WebRemoteTtsError, discover_remote_models, normalize_remote_endpoint, remote_client_python,
    resolve_remote_selection, select_remote_model, unload_audio_cpp_models,
)


def _status(message=None):
    return gr.update(value=message, visible=bool(message))


class RemoteModelPanel:
    def __init__(self):
        self.state = gr.State(RemoteTtsConfig())
        with gr.Accordion("Remote TTS server", open=False):
            self.endpoint = gr.Textbox(label="Remote server base URL", placeholder="http://127.0.0.1:8000")
            with gr.Row():
                self.refresh = gr.Button("Refresh remote models")
                self.unload = gr.Button("Unload audio.cpp models", interactive=False)
            self.model = gr.Dropdown(label="Remote speech model", choices=[], interactive=False)
            self.status = gr.Textbox(label="Remote server status", interactive=False, visible=False)
            with gr.Row():
                self.install = gr.Button("Install remote speech client", interactive=remote_client_install_supported())
                self.cancel_install = gr.Button("Cancel client installation", variant="stop", interactive=False)
            self.client_status = gr.Textbox(label="Remote client status", interactive=False, visible=False)
        self.selection_inputs = [self.endpoint, self.state]

    selection_for = staticmethod(resolve_remote_selection)

    @staticmethod
    def _display(config):
        choices = [(candidate.label, candidate.key) for candidate in config.candidates]
        return (
            config,
            gr.update(choices=choices, value=config.selected.key if config.selected else None, interactive=bool(choices)),
            gr.update(interactive=config.backend_kind is TtsBackendKind.AUDIO_CPP),
            _status(config.notice),
        )

    def bind(self, *, event_options):
        controls = [self.state, self.model, self.unload, self.status]
        active_installations: dict[str | None, Event] = {}
        installation_lock = Lock()

        def reset_on_url_change(endpoint, previous):
            try:
                endpoint = normalize_remote_endpoint(endpoint)
            except WebRemoteTtsError:
                endpoint = None
            if endpoint == previous.endpoint:
                return tuple(gr.skip() for _ in controls)
            notice = "Refresh remote models for this URL." if endpoint else ""
            return self._display(RemoteTtsConfig(notice=notice))

        def refresh_models(endpoint, previous):
            try:
                config = discover_remote_models(endpoint, previous)
            except WebRemoteTtsError as exc:
                return *self._display(RemoteTtsConfig(notice=str(exc))), gr.skip(), gr.skip()
            available = remote_client_python() is not None
            client = "Remote speech client is available." if available else "No installed remote speech client environment was found."
            return *self._display(config), _status(client), gr.update(
                interactive=remote_client_install_supported() and not available,
            )

        def choose_model(endpoint, config, key):
            try:
                return self._display(select_remote_model(endpoint, config, key))
            except WebRemoteTtsError as exc:
                return *[gr.skip()] * (len(controls) - 1), _status(str(exc))

        def unload_models(endpoint, config):
            try:
                unload_audio_cpp_models(endpoint, config)
            except WebRemoteTtsError as exc:
                return _status(str(exc))
            return _status("Unloaded the audio.cpp server’s models.")

        self.endpoint.input(
            reset_on_url_change, self.selection_inputs, controls, **event_options,
        )
        self.refresh.click(refresh_models, self.selection_inputs, [*controls, self.client_status, self.install], **event_options)
        self.model.input(choose_model, [*self.selection_inputs, self.model], controls, **event_options)
        self.unload.click(unload_models, self.selection_inputs, self.status, **event_options)

        def stop_installation(request: gr.Request):
            with installation_lock:
                if cancel := active_installations.get(request.session_hash):
                    cancel.set()

        def install_client(request: gr.Request, progress=gr.Progress()):
            cancel = Event()
            with installation_lock:
                active_installations[request.session_hash] = cancel
            available = remote_client_install_supported()
            try:
                yield (
                    _status("Preparing the remote speech client…"),
                    gr.update(interactive=False), gr.update(interactive=True),
                )
                try:
                    install_remote_client(on_progress=lambda description: progress(0, desc=description), cancel_event=cancel)
                except WebModelSetupError as exc:
                    message = str(exc)
                except Exception:
                    message = "Could not install the remote speech client."
                else:
                    message = "Remote speech client is ready."
                    available = False
            finally:
                with installation_lock:
                    active_installations.pop(request.session_hash, None)
            yield _status(message), gr.update(interactive=available), gr.update(interactive=False)

        self.install.click(
            install_client, outputs=[self.client_status, self.install, self.cancel_install], **event_options,
        )
        self.cancel_install.click(
            stop_installation, api_visibility="private", queue=False, show_progress="hidden",
        )
