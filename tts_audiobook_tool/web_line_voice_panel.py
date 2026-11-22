"""Browser controls for overriding the voice on selected book lines."""
from __future__ import annotations

import gradio as gr

from tts_audiobook_tool.web_backend import WebBackendError
from tts_audiobook_tool.web_line_voices import line_voice_rows, line_voice_sections, set_line_voice
from tts_audiobook_tool.web_voices import WebVoiceError, list_voices, load_voice


class LineVoicePanel:
    def __init__(self):
        with gr.Accordion("Assign voices to lines", open=False, visible=False) as self.panel:
            self.section = gr.Dropdown(label="Book section", choices=[], interactive=True)
            self.lines = gr.Dataframe(
                value=[], headers=["Line", "Manual voice", "Text"], datatype=["number", "str", "str"],
                label="Book lines", interactive=False, max_height=400, wrap=True, buttons=[],
            )
            with gr.Row():
                self.ranges = gr.Textbox(label="Line numbers", placeholder="2, 4-7, 12 or all")
                self.voice = gr.Dropdown(label="Saved voice", choices=[], interactive=True)
            with gr.Row():
                self.assign = gr.Button("Assign voice")
                self.clear = gr.Button("Clear manual voice")
            self.status = gr.Textbox(label="Line voice status", interactive=False, visible=False)
            self.workspace = gr.State(None)

    def bind(self, app: gr.Blocks, book: gr.State, *, event_options):
        def show_book(session, section, old_workspace):
            choices = line_voice_sections(session)
            workspace = (session.attribution_dir or session.project_dir) if session is not None else None
            same_book = workspace == old_workspace
            if not same_book or section not in {index for _, index in choices}:
                section = choices[0][1] if choices else None
            return (
                gr.update(visible=bool(choices)), gr.update(choices=choices, value=section),
                line_voice_rows(session, section), workspace,
                gr.skip() if same_book else "", gr.skip() if same_book else gr.update(value=None, visible=False),
            )

        def refresh_voices(selected=None):
            try:
                choices = [(voice.name, voice.voice_id) for voice in list_voices()]
            except WebVoiceError as error:
                return gr.skip(), gr.update(value=str(error), visible=True)
            selected = selected if selected in {voice_id for _, voice_id in choices} else None
            return gr.update(choices=choices, value=selected), gr.skip()

        def choose_line(event: gr.SelectData):
            row = event.row_value
            if row and type(row[0]) is int and row[0] > 0:
                return str(row[0])
            return gr.skip()

        def apply(session, section, ranges, selected=None, *, clear=False):
            try:
                if not clear and not selected:
                    raise WebVoiceError("Choose a saved voice to assign.")
                voice = None if clear else load_voice(selected).voice
                updated, changed = set_line_voice(session, ranges, voice)
            except (WebVoiceError, WebBackendError) as error:
                return gr.skip(), gr.skip(), gr.update(value=str(error), visible=True)
            count = f"{changed:,} {'line' if changed == 1 else 'lines'}"
            if not changed:
                message = "Those lines already use the selected assignment." if voice else "Those lines use automatic casting."
            elif voice:
                message = f"Assigned {voice.name} to {count}."
            else:
                message = f"Restored automatic casting for {count}."
            return updated, line_voice_rows(updated, section), gr.update(value=message, visible=True)

        book.change(
            show_book, [book, self.section, self.workspace],
            [self.panel, self.section, self.lines, self.workspace, self.ranges, self.status], **event_options,
        )
        self.section.input(line_voice_rows, [book, self.section], self.lines, **event_options)
        self.lines.select(choose_line, outputs=self.ranges, **event_options)
        voice_outputs = [self.voice, self.status]
        app.load(refresh_voices, outputs=voice_outputs, **event_options)
        gr.on([self.panel.expand, self.voice.focus], refresh_voices, self.voice, voice_outputs, **event_options)

        inputs = [book, self.section, self.ranges]
        outputs = [book, self.lines, self.status]
        self.assign.click(apply, [*inputs, self.voice], outputs, **event_options)
        self.clear.click(lambda session, section, ranges: apply(session, section, ranges, clear=True), inputs, outputs, **event_options)
