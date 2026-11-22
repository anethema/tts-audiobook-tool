import asyncio

import pytest

gr = pytest.importorskip("gradio", minversion="6.17.3")
from gradio.state_holder import SessionState

from tts_audiobook_tool.web_backend import close_book_session
from tts_audiobook_tool.web_text_panel import SectionTextPanel, TextImportPanel


def test_gradio_text_search_and_edit_target_the_loaded_line():
    events = {"api_visibility": "private"}
    with gr.Blocks(analytics_enabled=False) as app:
        book = gr.State(None)
        source = TextImportPanel()
        editor = SectionTextPanel()
        summary = gr.Textbox(interactive=False)
        source.bind(event_options=events)

        def import_text(*fields):
            session = source.import_book(*fields)
            return session, *editor.refresh(session)

        source.load.click(import_text, source.inputs, [book, *editor.refresh_outputs], **events)
        editor.bind(
            book, book_changed=lambda session: (session, session.summary()), changed_inputs=[],
            changed_outputs=[book, summary], event_options=events,
        )

    state = SessionState(app)

    async def trigger(component, kind, *values):
        fn = next(fn for fn in app.fns.values() if (component._id, kind) in fn.targets)
        result = await app.process_api(fn, list(values), state)
        return dict(zip(fn.outputs, result["data"], strict=True))

    async def exercise():
        visibility = await trigger(source.source, "input", "manual")
        assert visibility[source.pasted_text]["visible"] is True
        assert visibility[source.epub]["visible"] is False

        imported = await trigger(
            source.load, "click", "manual", None, None,
            "The rain fell. Rue closed the door.", "en", 40, "sentence", False,
        )
        assert imported[editor.line]["value"] == 0
        first_token = imported[editor.line_token]
        matching = await trigger(editor.search, "submit", None, 0, "CLOSED THE DOOR")
        assert matching[editor.line]["value"] == 1
        second_token = matching[editor.line_token]

        stale = await trigger(editor.apply, "click", None, 0, 1, first_token, "Rue locked the door.", None)
        assert stale[editor.status]["visible"] is True
        assert state[book._id].book.phrase_groups[1].text == "Rue closed the door."

        edited = await trigger(editor.apply, "click", None, 0, 1, second_token, "Rue locked the door.", None)
        assert edited[editor.text]["value"] == "Rue locked the door."
        assert state[book._id].book.phrase_groups[1].text == "Rue locked the door."
        assert edited[summary] == state[book._id].summary()

        titles = await trigger(
            editor.apply_titles, "click", None, 0, 1, edited[editor.title_token], "The Door", "Chapter One",
        )
        assert titles[editor.book_title] == "The Door"
        assert titles[editor.section_title] == "Chapter One"
        token = titles[editor.title_token]
        language = await trigger(editor.apply_language, "click", None, 0, 1, token, "")
        assert language[editor.project_language] == ""
        assert state[book._id].book.segmentation_settings.language_code == "en"

        pronunciations = await trigger(
            editor.apply_substitutions, "click", None, 0, 1, token,
            {"headers": ["Original word", "Read as"], "data": [["Rue", "Roo"]]},
        )
        assert pronunciations[editor.substitutions]["data"] == [["Rue", "Roo"]]
        assert state[book._id].word_substitutions == {"Rue": "Roo"}

        await trigger(editor.resegment, "click", None, 0, 1, token, "en", 40, "multi", False)
        assert state[book._id].book.segmentation_settings.strategy.id == "multi"
        assert state[book._id].language_code == ""

    try:
        asyncio.run(exercise())
    finally:
        close_book_session(state[book._id])


def test_gradio_can_split_and_rejoin_sections_with_the_same_line_selected():
    events = {"api_visibility": "private"}

    async def exercise():
        with gr.Blocks(analytics_enabled=False) as app:
            book = gr.State(None)
            source = TextImportPanel()
            editor = SectionTextPanel()

            def load():
                session = source.import_book(
                    "manual", None, None, "Rain fell. The road flooded. Rue went home.", "en", 40, "sentence", False,
                )
                return session, *editor.refresh(session)

            source.load.click(load, outputs=[book, *editor.refresh_outputs], **events)
            editor.bind(
                book, book_changed=lambda session: (session,), changed_inputs=[], changed_outputs=[book],
                event_options=events, source_epub=source.epub,
            )

        state = SessionState(app)

        async def trigger(component, kind, *values):
            fn = next(fn for fn in app.fns.values() if (component._id, kind) in fn.targets)
            result = await app.process_api(fn, list(values), state)
            return dict(zip(fn.outputs, result["data"], strict=True))

        try:
            initial = await trigger(source.load, "click")
            first_token = initial[editor.line_token]
            assert not initial[editor.split_section]["interactive"]
            selected = await trigger(editor.line, "input", None, 0, 1)
            assert selected[editor.split_section]["interactive"]
            cut = await trigger(editor.split_section, "click", None, 0, 1, selected[editor.line_token])
            assert cut[editor.section]["value"] == 1
            assert cut[editor.line]["value"] == 1
            assert state[book._id].book.section_start_indices() == [0, 1]
            assert cut[editor.join_previous_section]["interactive"]

            await trigger(editor.section, "input", None, 0)
            stale = await trigger(editor.join_next_section, "click", None, 0, 0, first_token)
            assert stale[editor.status]["visible"]
            assert state[book._id].book.section_start_indices() == [0, 1]
            selected = await trigger(editor.section, "input", None, 1)
            joined = await trigger(editor.join_previous_section, "click", None, 1, 1, selected[editor.line_token])
            assert joined[editor.section]["value"] == 0
            assert joined[editor.line]["value"] == 1
            assert state[book._id].book.section_start_indices() == [0]
            assert [group.text.strip() for group in state[book._id].book.phrase_groups] == [
                "Rain fell.", "The road flooded.", "Rue went home.",
            ]
        finally:
            close_book_session(state[book._id])

    asyncio.run(exercise())
