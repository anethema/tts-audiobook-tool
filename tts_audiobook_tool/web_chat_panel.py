"""Gradio Voice Chat controls and per-browser conversation state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock

import gradio as gr

from tts_audiobook_tool.web_api_keys import StoredApiKeyError, load_api_key
from tts_audiobook_tool.web_chat import (
    ChatState, ChatStopped, ChatTurn, WebChatError, default_system_prompt, prepare_chat_turn, transcribe_chat_input,
)
from tts_audiobook_tool.web_models import ModelRunError, render_model_sample
from tts_audiobook_tool.web_remote_models import REMOTE_MODEL_ID, WebRemoteTtsError
from tts_audiobook_tool.web_render import WebRenderError


def _status(message=None):
    return gr.update(value=message, visible=bool(message))


def _partial_messages(state, message, reply=None):
    messages = [*state.messages(), {"role": "user", "content": message}]
    if reply is not None:
        messages.append({"role": "assistant", "content": reply})
    return messages


class VoiceChatPanel:
    def __init__(self, installed_choices, default_model):
        self.state = gr.State(ChatState())
        self._active: dict[str | None, ChatTurn] = {}
        self._lock = Lock()

        self.history = gr.Chatbot(
            label="Conversation", height=370, render_markdown=False, allow_tags=False, buttons=["copy", "copy_all"],
        )
        self.message = gr.Textbox(label="Message", lines=3, placeholder="Type here or record a message below.")
        with gr.Accordion("Voice input", open=False):
            self.microphone = gr.Audio(
                sources=["microphone", "upload"], type="filepath", label="Record or upload a message",
                editable=False, buttons=[],
            )
            self.transcribe = gr.Button("Retry transcription", size="sm")
            self.transcription_status = gr.Textbox(label="Voice input status", interactive=False, visible=False)
        with gr.Row():
            self.send = gr.Button("Send", variant="primary")
            self.stop = gr.Button("Stop reply", variant="stop", interactive=False)
            self.clear = gr.Button("New chat")
            self.end = gr.Button("End chat")
        self.status = gr.Textbox(label="Chat status", interactive=False, visible=False)
        self.audio = gr.Audio(label="Latest reply audio", interactive=False, visible=False, buttons=["download"])
        with gr.Accordion("Chat settings", open=False):
            gr.Markdown("Set the LLM endpoint, model, and API key under Character voices in the Audiobook tab.")
            with gr.Row():
                self.echo = gr.Checkbox(label="Echo mode", info="Repeat your message as the reply.")
                self.stream = gr.Checkbox(value=True, label="Stream reply text")
            with gr.Row():
                self.speech_model = gr.Dropdown(
                    choices=self.model_choices(installed_choices), value=default_model or "",
                    label="Reply speech model", interactive=True,
                )
                self.language = gr.Textbox(value="en", label="Chat language code")
            self.system_prompt = gr.Textbox(value=default_system_prompt(), label="System prompt", lines=5)
            self.extra_params = gr.Textbox(label="Extra parameters for chat (JSON)", lines=3)

    @staticmethod
    def model_choices(installed_choices):
        return [("Text only", ""), *installed_choices]

    def bind(self, *, api_key, llm_model, effort, endpoint, voice_panel, model_settings,
             event_options, runtime_panel=None, remote_panel=None, console_for=None):
        outputs = [self.state, self.history, self.message, self.microphone, self.audio, self.status, self.send, self.stop]
        runtime_inputs = runtime_panel.inputs if runtime_panel is not None else ()
        remote_inputs = remote_panel.selection_inputs if remote_panel is not None else ()

        def stop_reply(request: gr.Request):
            with self._lock:
                active = self._active.get(request.session_hash)
            if active is not None:
                active.stop()

        def send_message(
            state, message, key, model, endpoint_url, reasoning, extra_json, prompt, echo, stream, speech_model,
            language, settings, request: gr.Request, *runtime_and_voice_fields,
        ):
            secret = (key or "").strip()
            try:
                if not echo and not secret:
                    secret = load_api_key(endpoint_url or "")
                turn = prepare_chat_turn(
                    state, message or "", api_key=secret, model=model or "", endpoint=endpoint_url or "",
                    effort=reasoning or "", extra_json=extra_json or "", system_prompt=prompt or "", echo=echo,
                    stream=stream,
                )
            except (StoredApiKeyError, WebChatError) as exc:
                yield *[gr.skip()] * 5, _status(str(exc)), gr.skip(), gr.skip()
                return

            with self._lock:
                self._active[request.session_hash] = turn
            try:
                yield (
                    gr.skip(), _partial_messages(state, turn.message), gr.update(interactive=False), gr.skip(),
                    gr.update(value=None, visible=False), _status("Preparing reply…"),
                    gr.update(interactive=False), gr.update(interactive=True),
                )
                reply = ""
                try:
                    for reply in turn.reply():
                        yield (
                            gr.skip(), _partial_messages(state, turn.message, reply), gr.skip(), gr.skip(), gr.skip(),
                            gr.skip(), gr.skip(), gr.skip(),
                        )
                    if turn.cancel_event.is_set():
                        raise ChatStopped("Reply stopped.")
                except WebChatError as exc:
                    yield (
                        gr.skip(), state.messages(), gr.update(interactive=True), gr.skip(), gr.skip(),
                        _status(str(exc)), gr.update(interactive=True), gr.update(interactive=False),
                    )
                    return

                completed = turn.commit(reply)
                audio = gr.skip()
                message = "Reply ready."
                if speech_model:
                    console = console_for(request.session_hash) if console_for is not None else None
                    if console is not None:
                        console("\nGenerating voice chat audio\n")
                    yield (
                        completed, completed.messages(), gr.update(value=""), None, gr.skip(),
                        _status("Generating reply audio…"), gr.skip(), gr.skip(),
                    )
                    try:
                        runtime_fields = runtime_and_voice_fields[:len(runtime_inputs)]
                        voice_start = len(runtime_inputs) + len(remote_inputs)
                        voice = voice_panel.snapshot(*runtime_and_voice_fields[voice_start:])
                        runtime = runtime_panel.snapshot(*runtime_fields) if runtime_panel is not None else None
                        remote = None
                        if remote_panel is not None and speech_model == REMOTE_MODEL_ID:
                            remote = remote_panel.selection_for(*runtime_and_voice_fields[len(runtime_inputs):voice_start])
                        source = Path(render_model_sample(
                            speech_model, reply, voice, language_code=language or "en", cancel_event=turn.cancel_event,
                            model_options=(settings or {}).get(speech_model),
                            runtime_settings=runtime,
                            **({"on_console": console} if console is not None else {}),
                            **({"remote_selection": remote} if remote is not None else {}),
                        ))
                        cached = None
                        try:
                            if turn.cancel_event.is_set():
                                raise ChatStopped("Reply audio stopped.")
                            cached = self.audio.move_resource_to_block_cache(source)
                            audio = gr.update(value=cached, visible=True)
                        finally:
                            if cached is None or Path(cached) != source:
                                source.unlink(missing_ok=True)
                    except ChatStopped as exc:
                        message = str(exc)
                    except (ModelRunError, WebRenderError, WebRemoteTtsError) as exc:
                        message = "Reply audio stopped." if turn.cancel_event.is_set() else str(exc)
                    except Exception:
                        message = "Could not generate reply audio."
                    if console is not None:
                        console(message + "\n")
                yield (
                    completed, completed.messages(), gr.update(value="", interactive=True), None, audio,
                    _status(message), gr.update(interactive=True), gr.update(interactive=False),
                )
            finally:
                with self._lock:
                    if self._active.get(request.session_hash) is turn:
                        self._active.pop(request.session_hash)

        gr.on(
            [self.send.click, self.message.submit], send_message,
            [
                self.state, self.message, api_key, llm_model, endpoint, effort, self.extra_params, self.system_prompt,
                self.echo, self.stream, self.speech_model, self.language, model_settings,
                *runtime_inputs, *remote_inputs, *voice_panel.inputs,
            ], outputs, **event_options,
        )

        def reset_chat():
            return (
                ChatState(), [], gr.update(value="", interactive=True), None, gr.update(value=None, visible=False),
                _status(), gr.update(interactive=True), gr.update(interactive=False), _status(),
            )

        def end_chat(state):
            return (
                replace(state, ended=True), state.messages(), gr.update(value="", interactive=False), None, gr.skip(),
                _status("Chat ended. Start a new chat to continue."),
                gr.update(interactive=False), gr.update(interactive=False), _status(),
            )

        stop_options = {"api_visibility": "private", "queue": False, "show_progress": "hidden"}
        self.stop.click(stop_reply, **stop_options)
        self.clear.click(stop_reply, **stop_options).then(
            reset_chat, outputs=[*outputs, self.transcription_status], **event_options,
        )
        self.end.click(stop_reply, **stop_options).then(
            end_chat, [self.state], [*outputs, self.transcription_status], **event_options,
        )

        def transcribe(path, language, state):
            if state.ended:
                return gr.skip(), _status("Start a new chat to record a message.")
            try:
                text = transcribe_chat_input(path, language)
            except WebChatError as exc:
                return gr.update(interactive=True), _status(str(exc))
            except Exception:
                return gr.update(interactive=True), _status("Could not transcribe the message.")
            return gr.update(value=text, interactive=True), _status()

        def transcribe_recording(path, language, state):
            if not state.ended:
                yield gr.update(value="", interactive=False), _status("Transcribing message…")
            yield transcribe(path, language, state)

        transcription_inputs = [self.microphone, self.language, self.state]
        transcription_outputs = [self.message, self.transcription_status]
        gr.on(
            [self.microphone.stop_recording, self.microphone.upload], transcribe_recording,
            transcription_inputs, transcription_outputs, trigger_mode="multiple", **event_options,
        )
        self.transcribe.click(transcribe, transcription_inputs, transcription_outputs, **event_options)
