"""Gradio controls for importing and editing an audiobook's text."""

from __future__ import annotations

from hashlib import sha256

import gradio as gr

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_DEFAULT, MAX_WORDS_PER_SEGMENT_MAX, MAX_WORDS_PER_SEGMENT_MIN
from tts_audiobook_tool.constants_config import PROJECT_DEFAULT_SEGMENTATION_STRATEGY
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.web_backend import BookSession, WebBackendError
from tts_audiobook_tool.web_text import (
    adjacent_section, can_join_line, change_line, change_section, import_book_text, reapply_browser_formatting,
    resegment_book, section_lines,
    set_project_language, set_titles, set_word_substitutions,
)


def _segmentation_controls(language_label):
    with gr.Row():
        language = gr.Textbox(value="en", label=language_label)
        max_words = gr.Number(
            value=MAX_WORDS_PER_SEGMENT_DEFAULT, precision=0, minimum=MAX_WORDS_PER_SEGMENT_MIN,
            maximum=MAX_WORDS_PER_SEGMENT_MAX, label="Maximum words per segment",
        )
    strategy = gr.Dropdown(
        choices=[(strategy.label, strategy.id) for strategy in SegmentationStrategy],
        value=PROJECT_DEFAULT_SEGMENTATION_STRATEGY.id, label="Segmentation strategy", interactive=True,
    )
    dialogue = gr.Checkbox(value=True, label="Split dialogue into separate lines")
    return language, max_words, strategy, dialogue


class TextImportPanel:
    def __init__(self):
        self.source = gr.Radio(
            choices=[("EPUB file", "epub"), ("TXT file", "plain_text"), ("Paste text", "manual")],
            value="epub", label="Book text source",
        )
        self.epub = gr.File(label="DRM-free EPUB", file_types=[".epub"], height=100)
        self.text_file = gr.File(label="UTF-8 TXT", file_types=[".txt"], height=100, visible=False)
        self.pasted_text = gr.Textbox(label="Book text", lines=8, visible=False)
        with gr.Accordion("Segmentation settings", open=False):
            self.language, self.max_words, self.strategy, self.dialogue = _segmentation_controls("Book language code")
        self.load = gr.Button("Import book text")
        self.settings_outputs = [self.language, self.max_words, self.strategy, self.dialogue]
        self.inputs = [self.source, self.epub, self.text_file, self.pasted_text, *self.settings_outputs]

    import_book = staticmethod(import_book_text)

    @staticmethod
    def restore_settings(session: BookSession) -> tuple:
        language, max_words, strategy, dialogue = session.book.segmentation_settings
        return language, max_words or MAX_WORDS_PER_SEGMENT_DEFAULT, strategy.id, dialogue

    def bind(self, *, event_options):
        def select_source(source):
            return tuple(gr.update(visible=source == choice) for choice in ("epub", "plain_text", "manual"))

        self.source.input(
            select_source, self.source, [self.epub, self.text_file, self.pasted_text],
            show_progress="hidden", **event_options,
        )


def _status(message=None):
    return gr.update(value=message, visible=bool(message))


def _excerpt(text: str) -> str:
    single_line = " ".join(text.split())
    return f"{single_line[:85]}…" if len(single_line) > 85 else single_line


class SectionTextPanel:
    def __init__(self):
        with gr.Accordion("Review and edit text", open=False, visible=False) as panel:
            self.section = gr.Dropdown(label="Book section", choices=[], interactive=True)
            with gr.Row():
                self.search = gr.Textbox(label="Find text in section", scale=3)
                self.find = gr.Button("Find", scale=1)
            self.line = gr.Dropdown(label="Text line", choices=[], interactive=True)
            self.text = gr.Textbox(label="Selected text", lines=4, interactive=False)
            with gr.Row():
                self.apply = gr.Button("Apply edit", interactive=False)
                self.delete = gr.Button("Delete line", interactive=False)
                self.join = gr.Button("Join next line", interactive=False)
            with gr.Row():
                self.boundary = gr.Dropdown(label="Split after phrase", choices=[], interactive=False, scale=3)
                self.split = gr.Button("Split line", interactive=False)
            with gr.Accordion("Book and section settings", open=False):
                with gr.Row():
                    self.book_title = gr.Textbox(label="Book title")
                    self.section_title = gr.Textbox(label="Selected section title")
                self.apply_titles = gr.Button("Apply titles", size="sm")
                self.title_token = gr.Textbox(visible=False)
                with gr.Row():
                    self.split_section = gr.Button("Start section at selected line", size="sm", interactive=False)
                    self.join_previous_section = gr.Button("Join previous section", size="sm", interactive=False)
                    self.join_next_section = gr.Button("Join next section", size="sm", interactive=False)
                with gr.Row():
                    self.project_language = gr.Textbox(label="Project language code", scale=3)
                    self.apply_language = gr.Button("Apply language", scale=1)
            with gr.Accordion("Resegment current text", open=False):
                self.segmentation = _segmentation_controls("Segmentation language code")
                gr.Markdown("Resegmentation can change the lines used by dialogue assignments and audiobook generation.")
                self.resegment = gr.Button("Resegment book")
            with gr.Accordion("Pronunciation substitutions", open=False):
                self.substitutions = gr.Dataframe(
                    headers=["Original word", "Read as"], datatype="str", type="array",
                    row_count=1, column_count=2, interactive=True,
                )
                self.apply_substitutions = gr.Button("Apply pronunciation substitutions", size="sm")
            with gr.Accordion("EPUB display formatting", open=False):
                self.format_source = gr.File(label="Replacement source EPUB (optional)", file_types=[".epub"])
                self.refresh_format = gr.Button("Reapply EPUB formatting", size="sm")
            self.status = gr.Textbox(label="Text edit status", interactive=False, visible=False)
            self.line_token = gr.Textbox(visible=False)
        self.panel = panel
        self.line_outputs = [
            self.text, self.apply, self.delete, self.join, self.boundary, self.split, self.line_token,
            self.split_section, self.join_previous_section, self.join_next_section,
        ]
        self.settings_outputs = [
            self.book_title, self.section_title, self.title_token, self.project_language, *self.segmentation,
            self.substitutions,
        ]
        self.refresh_outputs = [
            self.panel, self.section, self.search, self.line, *self.line_outputs, *self.settings_outputs, self.status,
        ]

    @staticmethod
    def _token(session, section, line, text):
        layout = tuple((item.title, len(item.phrase_groups)) for item in session.book.sections)
        source = repr((session.project_dir, session.attribution_dir, layout, section, line, text))
        return sha256(source.encode()).hexdigest()

    def _display_line(self, session, section, line):
        group = None
        if session is not None and section is not None and line is not None:
            first, groups = section_lines(session, section)
            if first <= line < first + len(groups):
                group = groups[line - first]
        available = group is not None
        can_delete = available and len(session.book.phrase_groups) > 1
        can_join = available and can_join_line(session, section, line)
        boundaries = (
            [(f"{index} · {_excerpt(phrase.text)}", index) for index, phrase in enumerate(group.phrases[:-1], 1)]
            if available else []
        )
        return (
            gr.update(value=group.text if available else "", interactive=available),
            gr.update(interactive=available), gr.update(interactive=can_delete), gr.update(interactive=can_join),
            gr.update(choices=boundaries, value=None, interactive=bool(boundaries)),
            gr.update(interactive=bool(boundaries)), self._token(session, section, line, group.text) if available else "",
            gr.update(interactive=available and line > first),
            gr.update(interactive=available and adjacent_section(session, section, -1) is not None),
            gr.update(interactive=available and adjacent_section(session, section, 1) is not None),
        )

    def _match_lines(self, session, section, query="", preferred=None):
        choices = []
        query = " ".join((query or "").casefold().split())
        if session is not None and section is not None:
            first, groups = section_lines(session, section)
            for line, group in enumerate(groups, first):
                if not query or query in " ".join(group.text.casefold().split()):
                    choices.append((f"{line + 1:,} · {_excerpt(group.text)}", line))
        selected = preferred if preferred in {line for _, line in choices} else choices[0][1] if choices else None
        return gr.update(choices=choices, value=selected), *self._display_line(session, section, selected)

    def _section_title_values(self, session, section):
        if session is None or section is None:
            return "", ""
        title = session.book.sections[section].title
        return title, self._token(session, section, None, f"{session.book.title}\0{title}")

    def _settings_values(self, session, section):
        section_title, token = self._section_title_values(session, section)
        if session is None:
            segmentation = ("en", MAX_WORDS_PER_SEGMENT_DEFAULT, PROJECT_DEFAULT_SEGMENTATION_STRATEGY.id, True)
            return "", section_title, token, "", *segmentation, []
        substitutions = [[word, replacement] for word, replacement in session.word_substitutions.items()]
        return (
            session.book.title, section_title, token, session.language_code,
            *TextImportPanel.restore_settings(session), substitutions,
        )

    def refresh(self, session: BookSession | None, section=None, line=None, message=None):
        choices = [
            (UtteranceUtil.section_label(session.book, index), index)
            for index, item in enumerate(session.book.sections) if item.phrase_groups
        ] if session is not None else []
        if section not in {index for _, index in choices}:
            section = choices[0][1] if choices else None
        return (
            gr.update(visible=bool(choices)), gr.update(choices=choices, value=section), "",
            *self._match_lines(session, section, preferred=line), *self._settings_values(session, section), _status(message),
        )

    def bind(self, book, *, book_changed, changed_inputs, changed_outputs, event_options, source_epub=None):
        self.section.input(
            lambda session, section: (
                "", *self._match_lines(session, section), *self._section_title_values(session, section), _status(),
            ),
            [book, self.section], [
                self.search, self.line, *self.line_outputs, self.section_title, self.title_token, self.status,
            ], **event_options,
        )

        def find_text(session, section, query):
            line, *detail = self._match_lines(session, section, query)
            return line, *detail, _status("No matching lines in this section." if not line["choices"] else None)

        gr.on(
            [self.find.click, self.search.submit], find_text, [book, self.section, self.search],
            [self.line, *self.line_outputs, self.status], **event_options,
        )
        self.line.input(
            lambda session, section, line: (*self._display_line(session, section, line), _status()),
            [book, self.section, self.line], [*self.line_outputs, self.status], **event_options,
        )

        outputs = [*changed_outputs, *self.refresh_outputs]

        def status_only(message):
            return *(gr.skip() for _ in outputs[:-1]), _status(message)

        def show_change(session, updated, section, line, context, message):
            if updated is session:
                return status_only("There are no changes to apply.")
            if len(updated.attributions.ranges) < len(session.attributions.ranges):
                message += " Some dialogue assignments need to be refreshed."
            next_section = min(section, len(updated.book.sections) - 1) if section is not None else None
            return *book_changed(updated, *context), *self.refresh(updated, next_section, line, message)

        def apply_to_book(session, section, line, change, context, message):
            if session is None:
                return status_only("Import book text first.")
            try:
                updated = change()
            except WebBackendError as exc:
                return status_only(str(exc))
            return show_change(session, updated, section, line, context, message)

        def check_selected_line(session, section, line, token):
            if session is None or section is None or line is None:
                raise WebBackendError("Choose a book section and line.")
            first, groups = section_lines(session, section)
            if type(line) is not int or not first <= line < first + len(groups):
                raise WebBackendError("Choose a text line in the current section.")
            if token != self._token(session, section, line, groups[line - first].text):
                raise WebBackendError("Select the current line again before applying an edit.")

        def apply_change(session, section, line, token, text, boundary, *context, operation):
            try:
                check_selected_line(session, section, line, token)
            except WebBackendError as exc:
                return status_only(str(exc))
            next_line = line + 1 if operation == "split" else line
            message = {
                "edit": "Text updated.", "split": "Line split.", "delete": "Line deleted.", "join": "Lines joined.",
            }[operation]
            return apply_to_book(
                session, section, next_line,
                lambda: change_line(session, section, line, operation, text=text or "", boundary=boundary),
                context, message,
            )

        inputs = [book, self.section, self.line, self.line_token, self.text, self.boundary, *changed_inputs]
        for operation, button in (("edit", self.apply), ("split", self.split), ("delete", self.delete), ("join", self.join)):
            button.click(
                lambda *values, action=operation: apply_change(*values, operation=action),
                inputs, outputs, **event_options,
            )

        def apply_section_change(session, section, line, token, *context, operation):
            try:
                check_selected_line(session, section, line, token)
                updated, next_section = change_section(session, section, line, operation)
            except WebBackendError as exc:
                return status_only(str(exc))
            message = "Started a new section at this line." if operation == "split" else "Sections joined."
            return show_change(session, updated, next_section, line, context, message)

        for operation, button in (
            ("split", self.split_section), ("join_previous", self.join_previous_section), ("join_next", self.join_next_section),
        ):
            button.click(
                lambda *values, action=operation: apply_section_change(*values, operation=action),
                [book, self.section, self.line, self.line_token, *changed_inputs], outputs, **event_options,
            )

        def bind_setting(button, fields, change, message):
            def apply_setting(session, section, line, token, *values):
                if session is None or section is None:
                    return status_only("Choose a book section.")
                if token != self._section_title_values(session, section)[1]:
                    return status_only("Select the current section again before applying settings.")
                return apply_to_book(
                    session, section, line, lambda: change(session, section, *values[:len(fields)]),
                    values[len(fields):], message,
                )

            button.click(
                apply_setting, [book, self.section, self.line, self.title_token, *fields, *changed_inputs],
                outputs, **event_options,
            )

        bind_setting(
            self.apply_titles, [self.book_title, self.section_title],
            lambda session, section, title, subtitle: set_titles(session, section, title or "", subtitle or ""),
            "Titles updated.",
        )
        bind_setting(
            self.apply_language, [self.project_language],
            lambda session, _, language: set_project_language(session, language or ""), "Project language updated.",
        )
        bind_setting(
            self.resegment, self.segmentation, lambda session, _, *settings: resegment_book(session, *settings),
            "Current text resegmented.",
        )
        bind_setting(
            self.apply_substitutions, [self.substitutions],
            lambda session, _, rows: set_word_substitutions(session, rows or []), "Pronunciation substitutions updated.",
        )

        def reapply_format(session, section, line, selected_source, *fields):
            if session is None:
                return status_only("Import book text first.")
            source, context = selected_source, fields
            if source_epub is not None:
                source = selected_source or fields[0]
                context = fields[1:]
            try:
                updated, message = reapply_browser_formatting(session, source)
            except WebBackendError as exc:
                return status_only(str(exc))
            return *book_changed(updated, *context), *self.refresh(updated, section, line, message)

        sources = [self.format_source] if source_epub is None else [self.format_source, source_epub]
        self.refresh_format.click(
            reapply_format, [book, self.section, self.line, *sources, *changed_inputs], outputs, **event_options,
        )
