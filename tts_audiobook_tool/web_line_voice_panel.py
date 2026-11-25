"""Browser controls for overriding the voice on selected book lines."""
from __future__ import annotations

from functools import partial

import gradio as gr

from tts_audiobook_tool.web_backend import WebBackendError
from tts_audiobook_tool.web_line_voices import (
    line_voice_page_count, line_voice_rows, line_voice_sections, set_line_voice,
)
from tts_audiobook_tool.web_voices import WebVoiceError, list_voices, load_voice


class LineVoicePanel:
    def __init__(self):
        with gr.Accordion("Assign voices to lines", open=False, visible=False) as self.panel:
            self.section = gr.Dropdown(label="Book section", choices=[], interactive=True)
            with gr.Row(visible=False) as self.navigation:
                self.previous_page = gr.Button("Previous page", size="sm")
                self.page = gr.Number(
                    value=1, label="Page", info="Press Enter to jump.", precision=0, step=1, interactive=True,
                )
                self.next_page = gr.Button("Next page", size="sm")
            self.lines = gr.Radio(
                label="Book lines", choices=[], value=None, interactive=True, elem_id="line-voice-picker",
            )
            with gr.Row():
                self.ranges = gr.Textbox(label="Line numbers", placeholder="2, 4-7, 12 or all for the entire book")
                self.voice = gr.Dropdown(label="Saved voice", choices=[], interactive=True)
            with gr.Row():
                self.assign = gr.Button("Assign voice")
                self.clear = gr.Button("Clear manual voice")
            self.status = gr.Textbox(label="Line voice status", interactive=False, visible=False)
            self.workspace = gr.State(None)
            self.is_open = gr.State(False)

    def bind(self, app: gr.Blocks, book: gr.State, *, event_options):
        def show_page(session, section, page=1, *, offset=0):
            pages = line_voice_page_count(session, section)
            page = max(1, min((page or 1) + offset, pages))
            choices = [
                (f"{number:,} · {voice} · {text}", number)
                for number, voice, text in line_voice_rows(session, section, page)
            ]
            return {
                self.lines: gr.update(choices=choices, value=None),
                self.page: gr.update(value=page, label=f"Page (of {pages:,})" if pages else "Page"),
                self.navigation: gr.update(visible=pages > 1),
                self.previous_page: gr.update(interactive=page > 1),
                self.next_page: gr.update(interactive=page < pages),
            }

        def show_section_choices(session, section, page):
            choices = line_voice_sections(session)
            if section not in {index for _, index in choices}:
                section, page = (choices[0][1] if choices else None), 1
            return {
                self.section: gr.update(choices=choices, value=section),
                **show_page(session, section, page),
            }

        def show_book(session, section, page, old_workspace, is_open):
            workspace = (session.attribution_dir or session.project_dir) if session is not None else None
            has_lines = session is not None and bool(session.book.phrase_groups)
            updates = {self.workspace: workspace, self.panel: gr.update(visible=has_lines)}
            if workspace != old_workspace:
                section, page = None, 1
                updates.update({
                    self.ranges: "", self.status: gr.update(value=None, visible=False),
                    self.section: gr.update(choices=[], value=None), self.page: gr.update(value=1),
                })
            if not has_lines:
                updates.update({
                    self.panel: gr.update(visible=False, open=False), self.is_open: False,
                    self.lines: gr.update(choices=[], value=None),
                })
            elif is_open:
                updates.update(show_section_choices(session, section, page))
            return updates

        def open_panel(session, section, page):
            return {self.is_open: True, **show_section_choices(session, section, page)}

        def refresh_voices(selected=None):
            try:
                choices = [(voice.name, voice.voice_id) for voice in list_voices()]
            except WebVoiceError as error:
                return gr.skip(), gr.update(value=str(error), visible=True)
            selected = selected if selected in {voice_id for _, voice_id in choices} else None
            return gr.update(choices=choices, value=selected), gr.skip()

        def choose_line(number):
            return str(number) if number is not None else gr.skip()

        def apply(session, ranges, selected=None, *, clear=False):
            try:
                if not clear and not selected:
                    raise WebVoiceError("Choose a saved voice to assign.")
                voice = None if clear else load_voice(selected).voice
                updated, changed = set_line_voice(session, ranges, voice)
            except (WebVoiceError, WebBackendError) as error:
                return gr.skip(), gr.update(value=str(error), visible=True)
            count = f"{changed:,} {'line' if changed == 1 else 'lines'}"
            if not changed:
                message = "Those lines already use the selected assignment." if voice else "Those lines use automatic casting."
            elif voice:
                message = f"Assigned {voice.name} to {count}."
            else:
                message = f"Restored automatic casting for {count}."
            return updated if changed else gr.skip(), gr.update(value=message, visible=True)

        page_inputs = [book, self.section, self.page]
        page_outputs = [self.lines, self.page, self.navigation, self.previous_page, self.next_page]
        section_outputs = [self.section, *page_outputs]
        book.change(
            show_book, [*page_inputs, self.workspace, self.is_open],
            [self.panel, self.workspace, self.is_open, self.ranges, self.status, *section_outputs], **event_options,
        )
        self.panel.expand(open_panel, page_inputs, [self.is_open, *section_outputs], **event_options)
        self.panel.collapse(
            lambda: (False, gr.update(choices=[], value=None)), outputs=[self.is_open, self.lines], **event_options,
        )
        self.section.input(show_page, [book, self.section], page_outputs, **event_options)
        self.page.submit(show_page, page_inputs, page_outputs, **event_options)
        self.previous_page.click(partial(show_page, offset=-1), page_inputs, page_outputs, **event_options)
        self.next_page.click(partial(show_page, offset=1), page_inputs, page_outputs, **event_options)
        self.lines.input(choose_line, self.lines, self.ranges, **event_options)
        voice_outputs = [self.voice, self.status]
        app.load(refresh_voices, outputs=voice_outputs, **event_options)
        gr.on([self.panel.expand, self.voice.focus], refresh_voices, self.voice, voice_outputs, **event_options)

        # A successful edit triggers book.change, which refreshes the visible page once.
        outputs = [book, self.status]
        self.assign.click(apply, [book, self.ranges, self.voice], outputs, **event_options)
        self.clear.click(partial(apply, clear=True), [book, self.ranges], outputs, **event_options)
