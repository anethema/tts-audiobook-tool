from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from contextlib import contextmanager
from threading import Event, Lock

import gradio as gr

from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_DEFAULT, PROJECT_BATCH_SIZE_MAX
from tts_audiobook_tool.l import L
from tts_audiobook_tool.web_backend import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    BookSession,
    WebBackendError,
    attribute_dialog,
    character_rows,
    close_book_session,
    identify_characters,
    import_epub,
)
from tts_audiobook_tool.web_dialogue import correct_speaker, dialogue_rows, dialogue_sections, speaker_choices
from tts_audiobook_tool.web_projects import (
    WebProjectError,
    create_project,
    list_projects,
    load_project,
    save_project,
)
from tts_audiobook_tool.web_legacy_project import (
    WebLegacyProjectError, list_console_projects, load_console_project,
)
from tts_audiobook_tool.web_models import (
    ModelRunError,
    installed_model_choices,
    render_model_book,
    render_model_sample,
    transcribe_model_reference,
)
from tts_audiobook_tool.web_render import RenderVoice


TITLE = "TTS Audiobook Tool"
PRIVATE_EVENT = {"api_visibility": "private", "concurrency_id": "local-work"}
REASONING_EFFORTS = [
    ("Model default", ""),
    ("None", "none"),
    ("Minimal", "minimal"),
    ("Low", "low"),
    ("Medium", "medium"),
    ("High", "high"),
    ("Extra high", "xhigh"),
    ("Maximum", "max"),
]
CSS = """
.gradio-container { max-width: 1260px !important; }
#app-heading { margin-bottom: 12px; }
#app-heading h1 { letter-spacing: -0.045em; }
#voice-settings { border: 1px solid var(--border-color-primary);
                  border-radius: var(--radius-lg); padding: 16px; }
#voice-settings h2 { font-size: 1.1rem; margin: 0; }
#sample-players > div { flex: 1 1 275px; }
"""


def _make_voice(reference: str | None, transcript: str, instructions: str) -> RenderVoice:
    return RenderVoice(reference, (transcript or "").strip(), (instructions or "").strip())


def _book_summary(session: BookSession) -> str:
    lines = [session.book.title, session.summary()]
    titles = [section.title for section in session.book.sections if section.title]
    if titles:
        lines.extend(["", *titles])
    if session.warnings:
        lines.extend(["", "Import warnings", *session.warnings])
    return "\n".join(lines)


def _review_components(
    session: BookSession | None,
    voices: dict[str, RenderVoice],
    role: str | None = None,
) -> tuple[dict, dict]:
    rows = character_rows(session) if session is not None and session.roster is not None else []
    narrator = session.roster.narrator if session is not None and session.roster is not None else None
    narrator_ids = {"NARRATOR"}
    if narrator is not None:
        narrator_ids.add(narrator.id)

    choices = [(row.name, row.speaker_id) for row in rows if row.speaker_id not in narrator_ids]
    selected = role if role in {speaker_id for _, speaker_id in choices} else None
    table = []
    for row in rows:
        if row.speaker_id in narrator_ids:
            voice_label = "Voice settings"
        elif row.speaker_id in voices:
            voice_label = "Custom voice"
        else:
            voice_label = "Narrator"
        table.append([row.name, row.line_count, row.voice_hint, voice_label])
    return (
        gr.update(value=table, visible=bool(rows)),
        gr.update(choices=choices, value=selected, visible=bool(choices)),
    )


def build_app() -> gr.Blocks:
    """Build the loopback UI without loading models or opening a connection."""
    if not hasattr(L, "logger"):
        L.init("tts-audiobook-web", level=logging.WARNING)
    model_choices = installed_model_choices()
    model_labels = {model_id: label for label, model_id in model_choices}
    default_model = next(iter(model_labels), None)
    running_jobs: dict[str | None, Event] = {}
    running_jobs_lock = Lock()

    @contextmanager
    def running_job(request: gr.Request):
        with running_jobs_lock:
            cancel = running_jobs[request.session_hash] = Event()
        try:
            yield cancel
        finally:
            with running_jobs_lock:
                running_jobs.pop(request.session_hash, None)

    def stop_generation(request: gr.Request):
        with running_jobs_lock:
            if cancel := running_jobs.get(request.session_hash):
                cancel.set()

    with gr.Blocks(title=TITLE, analytics_enabled=False, delete_cache=(3600, 86400)) as app:
        book = gr.State(None, delete_callback=close_book_session)
        cast = gr.State({})
        active_project = gr.State(None)
        gr.Markdown(f"# {TITLE}", elem_id="app-heading")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=275, elem_id="voice-settings"):
                gr.Markdown("## Voice settings")
                reference = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Reference audio",
                    buttons=[],
                )
                gr.Markdown("Used for the narrator and for all text samples.")
                with gr.Accordion("Transcript and delivery instructions", open=False):
                    transcript = gr.Textbox(
                        label="Reference transcript",
                        placeholder="What is spoken in the reference recording",
                        lines=2,
                    )
                    with gr.Row():
                        reference_language = gr.Textbox(value="en", label="Reference language code")
                        transcribe_reference = gr.Button("Transcribe reference")
                    transcript_status = gr.Textbox(label="Transcription status", interactive=False, visible=False)
                    instructions = gr.Textbox(
                        label="Delivery instructions",
                        placeholder="For example, warm and unhurried",
                        lines=2,
                    )
                    gr.Markdown("Transcript and delivery controls depend on the speech model.")

            with gr.Column(scale=3, min_width=345):
                with gr.Tabs():
                    with gr.Tab("EPUB Audiobook"):
                        with gr.Group():
                            epub = gr.File(label="DRM-free EPUB", file_types=[".epub"], height=100)
                            book_language = gr.Textbox(value="en", label="Book language code")
                            load = gr.Button("Import EPUB")
                            summary = gr.Textbox(
                                label="Book summary", interactive=False, lines=4, max_lines=12, visible=False
                            )

                        with gr.Accordion("Saved projects", open=False):
                            with gr.Row():
                                saved_projects = gr.Dropdown(
                                    label="Saved project", choices=[], interactive=True, scale=3
                                )
                                open_project = gr.Button("Open project", interactive=False)
                            with gr.Row(visible=False) as console_projects_row:
                                console_projects = gr.Dropdown(
                                    label="Console project", choices=[], interactive=True, scale=3
                                )
                                copy_console = gr.Button("Copy from console")
                            with gr.Row():
                                project_name = gr.Textbox(label="New project name", scale=3)
                                save_new = gr.Button("Save new project", interactive=False)
                            save_changes = gr.Button("Save changes", interactive=False)
                            project_status = gr.Textbox(label="Project status", interactive=False, visible=False)

                        with gr.Accordion("Character voices", open=False):
                            gr.Markdown(
                                "Character identification and dialog attribution send book text to OpenAI "
                                "when you click their buttons. Audiobook synthesis runs locally."
                            )
                            api_key = gr.Textbox(
                                label="OpenAI API key", type="password",
                                html_attributes=gr.InputHTMLAttributes(autocomplete="off"),
                            )
                            with gr.Row():
                                llm_model = gr.Textbox(value=DEFAULT_OPENAI_MODEL, label="OpenAI model")
                                effort = gr.Dropdown(
                                    REASONING_EFFORTS, value=DEFAULT_REASONING_EFFORT, label="Reasoning effort",
                                    info="Available levels vary by OpenAI model.", interactive=True,
                                )
                            with gr.Row():
                                identify = gr.Button("Identify characters", interactive=False)
                                attribute = gr.Button("Assign dialog", interactive=False)
                            analysis_status = gr.Textbox(label="Analysis status", interactive=False, visible=False)
                            characters = gr.Dataframe(
                                headers=["Character", "Dialog lines", "Voice suggestion", "Voice source"],
                                datatype=["str", "number", "str", "str"],
                                interactive=False,
                                max_height=340,
                                visible=False,
                            )
                            cast_role = gr.Dropdown(
                                label="Character to cast", choices=[], interactive=True, visible=False
                            )
                            with gr.Group(visible=False) as cast_editor:
                                cast_reference = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="Character reference audio",
                                    buttons=[],
                                )
                                with gr.Accordion("Character transcript and delivery instructions", open=False):
                                    cast_transcript = gr.Textbox(label="Reference transcript", lines=2)
                                    transcribe_cast = gr.Button("Transcribe character reference")
                                    cast_instructions = gr.Textbox(label="Delivery instructions", lines=2)
                                with gr.Row():
                                    save_cast = gr.Button("Save character voice")
                                    clear_cast = gr.Button("Use narrator voice")
                                cast_status = gr.Textbox(label="Casting status", interactive=False, visible=False)

                            with gr.Accordion("Review dialogue assignments", open=False, visible=False) as review_panel:
                                review_section = gr.Dropdown(label="Book section", choices=[], interactive=True)
                                review_lines = gr.Dataframe(
                                    headers=["Line", "Dialogue", "Speaker", "Uncertain"],
                                    datatype=["number", "str", "str", "bool"],
                                    column_widths=[75, "55%", "23%", 85],
                                    interactive=False,
                                    max_height=360,
                                    visible=False,
                                )
                                review_line = gr.Dropdown(
                                    label="Dialogue line to correct", choices=[], interactive=True, visible=False
                                )
                                with gr.Row():
                                    review_speaker = gr.Dropdown(
                                        label="New speaker", choices=[], interactive=True, visible=False
                                    )
                                    save_speaker = gr.Button("Save correction", visible=False)
                                review_status = gr.Textbox(label="Dialogue review status", interactive=False, visible=False)
                                gr.Markdown("For a saved project, use **Save changes** to keep your corrections.")

                        with gr.Group():
                            book_model = gr.Dropdown(
                                choices=model_choices, value=default_model, label="Speech model", interactive=True
                            )
                            with gr.Accordion("Generation settings", open=False):
                                validate_audio = gr.Checkbox(value=True, label="Validate audio with Whisper")
                                batch_size = gr.Number(
                                    value=PROJECT_BATCH_SIZE_DEFAULT, precision=0, minimum=1,
                                    maximum=PROJECT_BATCH_SIZE_MAX, label="Batch size",
                                )
                            with gr.Row():
                                generate_book = gr.Button(
                                    "Generate or resume audiobook", variant="primary", interactive=False,
                                )
                                stop_book = gr.Button("Stop generation", variant="stop", interactive=False)
                            book_status = gr.Textbox(label="Audiobook status", interactive=False, visible=False)
                            audiobook = gr.File(label="Download audiobook", interactive=False, visible=False)

                    with gr.Tab("Text Sample"):
                        sample_text = gr.Textbox(
                            label="Text to synthesize",
                            placeholder="Enter a short passage to hear how it sounds.",
                            lines=5,
                        )
                        sample_language = gr.Textbox(value="en", label="Sample language code")
                        sample_models = gr.Dropdown(
                            choices=model_choices,
                            value=[default_model] if default_model else [],
                            label="Speech models",
                            info="Select one or more models to render the same passage.",
                            multiselect=True,
                            interactive=True,
                        )
                        with gr.Row():
                            generate_samples = gr.Button(
                                "Generate samples", variant="primary", interactive=bool(model_choices)
                            )
                            stop_samples = gr.Button("Stop generation", variant="stop", interactive=False)
                        sample_status = gr.Textbox(
                            value=None if model_choices else "No installed speech models were found.",
                            label="Sample status",
                            interactive=False,
                            visible=not model_choices,
                        )
                        with gr.Row(elem_id="sample-players"):
                            sample_players = [
                                gr.Audio(interactive=False, visible=False, buttons=["download"])
                                for _ in model_choices
                            ]

        def present_book(session, voices, problem=None):
            table, roles = _review_components(session, voices)
            return (
                session,
                voices,
                gr.update(value=_book_summary(session) if session else problem, visible=bool(session or problem)),
                table,
                roles,
                gr.update(visible=False),
                gr.update(value=None, visible=False),
                gr.update(interactive=session is not None),
                gr.update(interactive=session is not None and session.roster is not None),
                gr.update(interactive=session is not None and bool(model_choices)),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        def load_book(path: str | None, language: str, previous: BookSession | None):
            session = None
            problem = None
            if path is None:
                problem = "Choose an EPUB to import."
            else:
                try:
                    session = import_epub(path, language_code=language or "")
                except WebBackendError as exc:
                    problem = str(exc)
                except Exception:
                    problem = "Could not import the EPUB."
            close_book_session(previous)
            return (
                *present_book(session, {}, problem),
                None, gr.update(interactive=session is not None), gr.update(interactive=False),
                gr.update(value=None, visible=False),
            )

        def invalidate_book(previous: BookSession | None):
            close_book_session(previous)
            return (
                *present_book(None, {}),
                None, gr.update(interactive=False), gr.update(interactive=False), gr.update(value=None, visible=False),
            )

        book_outputs = [
            book, cast, summary, characters, cast_role, cast_editor, analysis_status,
            identify, attribute, generate_book, book_status, audiobook,
        ]
        project_actions = [active_project, save_new, save_changes, project_status]
        import_outputs = [*book_outputs, *project_actions]
        replaced_upload = gr.on([epub.upload, epub.clear], invalidate_book, [book], import_outputs, **PRIVATE_EVENT)
        imported_book = load.click(load_book, [epub, book_language, book], import_outputs, **PRIVATE_EVENT)

        def refresh_projects(selected=None):
            choices = [(project.name, project.project_id) for project in list_projects()]
            return (
                gr.update(choices=choices, value=selected),
                gr.update(interactive=bool(choices)),
            )

        def list_saved_projects():
            try:
                return *refresh_projects(), gr.skip()
            except WebProjectError as exc:
                return gr.skip(), gr.skip(), gr.update(value=str(exc), visible=True)

        def discover_console_projects():
            choices = [(item.name, item.project_id) for item in list_console_projects()]
            return gr.update(visible=bool(choices)), gr.update(
                choices=choices, value=choices[0][1] if choices else None,
            )

        def open_project_snapshot(project_id, previous, *, from_console=False):
            loaded = None
            try:
                if from_console:
                    loaded = load_console_project(project_id or "", model_choices=model_choices)
                    name, saved_id = loaded.name, None
                else:
                    loaded = load_project(project_id or "")
                    name, saved_id = loaded.summary.name, loaded.summary.project_id
                source = epub.move_resource_to_block_cache(loaded.source_epub)
                voice_reference = reference.move_resource_to_block_cache(loaded.narrator.reference_path)
            except (WebProjectError, WebLegacyProjectError) as exc:
                message = str(exc)
            except Exception:
                message = "Could not open the project."
            else:
                close_book_session(previous)
                installed = loaded.model_id in model_labels
                message = f"Copied {name} from the console. Save it as a new browser project." if from_console else f"Opened {name}."
                if not installed:
                    message += " Choose an installed speech model to generate audio."
                return (
                    *present_book(loaded.session, loaded.voices),
                    saved_id, gr.update(interactive=True), gr.update(interactive=saved_id is not None),
                    gr.update(value=message, visible=True),
                    source,
                    loaded.session.book.segmentation_settings.language_code or "en",
                    voice_reference, loaded.narrator.transcript, loaded.narrator.instructions,
                    loaded.model_id if installed else None,
                    loaded.validate_audio, loaded.batch_size, name,
                )
            if loaded is not None:
                close_book_session(loaded.session)
            return (
                *[gr.skip()] * (len(import_outputs) - 1),
                gr.update(value=message, visible=True),
                *[gr.skip()] * 9,
            )

        def open_console_snapshot(project_id, previous):
            return open_project_snapshot(project_id, previous, from_console=True)

        def store_project(
            operation, destination, session, voices, model, source, ref, transcription, direction, validate, batch,
        ):
            try:
                if session is None:
                    raise WebProjectError("Import an EPUB first.")
                saved = operation(
                    destination or "", session, narrator=_make_voice(ref, transcription, direction),
                    voices=voices, model_id=model or "", source_epub=source,
                    validate_audio=validate, batch_size=batch or PROJECT_BATCH_SIZE_DEFAULT,
                )
            except WebProjectError as exc:
                message = str(exc)
            except Exception:
                message = "Could not save the project."
            else:
                try:
                    choices, open_button = refresh_projects(saved.project_id)
                    snapshot = load_project(saved.project_id)
                except WebProjectError:
                    message = "Project saved. Reload the page to reopen it."
                else:
                    close_book_session(session)
                    return (
                        snapshot.session, snapshot.voices, saved.project_id, choices, open_button,
                        gr.update(interactive=True), gr.update(value=f"Saved {saved.name}.", visible=True),
                    )
            return (*[gr.skip()] * 6, gr.update(value=message, visible=True))

        def save_named_project(destination, *values):
            return store_project(create_project, destination, *values)

        def save_loaded_project(destination, *values):
            return store_project(save_project, destination, *values)

        app.load(list_saved_projects, outputs=[saved_projects, open_project, project_status], **PRIVATE_EVENT)
        app.load(discover_console_projects, outputs=[console_projects_row, console_projects], **PRIVATE_EVENT)
        opened_project_outputs = [
            *import_outputs, epub, book_language, reference, transcript, instructions, book_model,
            validate_audio, batch_size, project_name,
        ]
        opened_project = open_project.click(
            open_project_snapshot, [saved_projects, book], opened_project_outputs, **PRIVATE_EVENT,
        )
        copied_console = copy_console.click(
            open_console_snapshot, [console_projects, book], opened_project_outputs, **PRIVATE_EVENT,
        )
        persist_inputs = [
            book, cast, book_model, epub, reference, transcript, instructions, validate_audio, batch_size,
        ]
        persist_outputs = [book, cast, active_project, saved_projects, open_project, save_changes, project_status]
        save_new.click(save_named_project, [project_name, *persist_inputs], persist_outputs, **PRIVATE_EVENT)
        save_changes.click(save_loaded_project, [active_project, *persist_inputs], persist_outputs, **PRIVATE_EVENT)

        def analyze_book(
            operation: Callable,
            session: BookSession | None,
            key: str | None,
            model: str,
            reasoning: str,
            voices: dict[str, RenderVoice],
            role: str | None,
            *,
            new_roster: bool = False,
            **kwargs,
        ):
            secret = (key or "").strip()
            try:
                if session is None:
                    raise WebBackendError("Import an EPUB first.")
                if not secret:
                    raise WebBackendError("Enter an OpenAI API key.")
                session = operation(session, secret, model, effort=reasoning, **kwargs)
            except WebBackendError as exc:
                message = str(exc).replace(secret, "[redacted]") if secret else str(exc)
            except Exception:
                message = "OpenAI analysis failed. Check your settings and try again."
            else:
                if new_roster:
                    voices = {}
                    role = None
                table, roles = _review_components(session, voices, role)
                message = session.attribution_notice or (
                    "Character roster ready. Assign dialog next." if new_roster else "Dialog assignments ready."
                )
                return (
                    session, voices, _book_summary(session), table, roles,
                    gr.update(value=message, visible=True),
                    gr.update(interactive=session.roster is not None),
                )
            return (*[gr.skip()] * 5, gr.update(value=message, visible=True), gr.skip())

        def find_characters(session, key, model, reasoning, voices, role, progress=gr.Progress()):
            progress(0, desc="Identifying characters")
            return analyze_book(identify_characters, session, key, model, reasoning, voices, role, new_roster=True)

        def find_speakers(session, key, model, reasoning, voices, role, progress=gr.Progress()):
            progress(0, desc="Assigning dialog")
            return analyze_book(
                attribute_dialog, session, key, model, reasoning, voices, role,
                on_progress=lambda done, total: progress((done, total), desc="Assigning dialog"),
            )

        analysis_inputs = [book, api_key, llm_model, effort, cast, cast_role]
        analysis_outputs = [book, cast, summary, characters, cast_role, analysis_status, attribute]
        identified = identify.click(find_characters, analysis_inputs, analysis_outputs, **PRIVATE_EVENT)
        attributed = attribute.click(find_speakers, analysis_inputs, analysis_outputs, **PRIVATE_EVENT)

        def show_voice(role: str | None, voices: dict[str, RenderVoice]):
            voice = voices.get(role, RenderVoice())
            return (
                gr.update(visible=role is not None),
                cast_reference.move_resource_to_block_cache(voice.reference_path),
                voice.transcript, voice.instructions,
                gr.update(value=None, visible=False),
            )

        cast_fields = [cast_editor, cast_reference, cast_transcript, cast_instructions, cast_status]
        cast_role.change(show_voice, [cast_role, cast], cast_fields, **PRIVATE_EVENT)

        def update_cast(session, voices, role, ref=None, transcription="", direction=""):
            if session is None or role is None or session.roster is None or role not in session.roster.ids:
                return (*[gr.skip()] * 5, gr.update(
                    value="Choose a character from the roster.", visible=True
                ))
            next_voices = dict(voices)
            voice = _make_voice(ref, transcription, direction)
            if voice == RenderVoice():
                next_voices.pop(role, None)
                message = "This character will use the narrator voice."
            else:
                next_voices[role] = voice
                message = "Character voice saved."
            table, _ = _review_components(session, next_voices, role)
            return (
                next_voices, table, voice.reference_path, voice.transcript, voice.instructions,
                gr.update(value=message, visible=True),
            )

        cast_inputs = [book, cast, cast_role]
        cast_outputs = [cast, characters, cast_reference, cast_transcript, cast_instructions, cast_status]
        save_cast.click(
            update_cast,
            [*cast_inputs, cast_reference, cast_transcript, cast_instructions],
            cast_outputs,
            **PRIVATE_EVENT,
        )
        clear_cast.click(update_cast, cast_inputs, cast_outputs, **PRIVATE_EVENT)

        def transcribe_voice(path: str | None, language: str):
            try:
                if path is None:
                    raise ModelRunError("Upload or record a reference clip first.")
                text = transcribe_model_reference(path, language_code=language or "en")
            except ModelRunError as exc:
                message = str(exc)
            except Exception:
                message = "Could not transcribe the reference audio."
            else:
                return text, gr.update(value=None, visible=False)
            return gr.skip(), gr.update(value=message, visible=True)

        transcribe_reference.click(
            transcribe_voice, [reference, reference_language], [transcript, transcript_status], **PRIVATE_EVENT,
        )
        transcribe_cast.click(
            transcribe_voice, [cast_reference, book_language], [cast_transcript, cast_status], **PRIVATE_EVENT,
        )

        def update_review_sections(session, section=None):
            choices = dialogue_sections(session)
            selected = section if section in {index for _, index in choices} else None
            return gr.update(visible=bool(choices)), gr.update(choices=choices, value=selected)

        def section_controls(session, section, selected_line=None):
            rows = (
                dialogue_rows(session, section)
                if section in {index for _, index in dialogue_sections(session)}
                else []
            )
            choices = [
                (f"{row.first_line + 1} · {' '.join(row.text.split())[:75]}", row.first_line)
                for row in rows
            ]
            return (
                gr.update(
                    value=[[row.first_line + 1, row.text, row.speaker, row.uncertain] for row in rows],
                    visible=bool(rows),
                ),
                gr.update(choices=choices, value=selected_line, visible=bool(rows)),
                gr.update(choices=speaker_choices(session) if rows else [], value=None, visible=bool(rows)),
                gr.update(visible=bool(rows)),
            )

        def display_review_section(session, section):
            return (*section_controls(session, section), gr.update(value=None, visible=False))

        def update_dialogue_speaker(session, voices, role, section, line, speaker):
            try:
                if session is None:
                    raise WebBackendError("Import an EPUB first.")
                session = correct_speaker(session, section, line, speaker)
            except WebBackendError as exc:
                return (*[gr.skip()] * 6, gr.update(value=str(exc), visible=True))
            table, _ = _review_components(session, voices, role)
            return (
                session, table, *section_controls(session, section, line),
                gr.update(value="Speaker assignment updated.", visible=True),
            )

        for event in (replaced_upload, imported_book, opened_project, copied_console):
            event.then(
                update_review_sections, [book], [review_panel, review_section],
                show_progress="hidden", **PRIVATE_EVENT,
            )
        for event in (identified, attributed):
            event.then(
                update_review_sections, [book, review_section], [review_panel, review_section],
                show_progress="hidden", **PRIVATE_EVENT,
            )
        review_outputs = [review_lines, review_line, review_speaker, save_speaker, review_status]
        review_section.change(display_review_section, [book, review_section], review_outputs, **PRIVATE_EVENT)
        save_speaker.click(
            update_dialogue_speaker, [book, cast, cast_role, review_section, review_line, review_speaker],
            [book, characters, *review_outputs], **PRIVATE_EVENT,
        )

        def generate_audiobook(
            session, voices, model, ref, transcription, direction, validate, batch, request: gr.Request,
            progress=gr.Progress(),
        ):
            if session is None or model not in model_labels:
                yield (
                    gr.update(value="Import an EPUB and choose an installed speech model.", visible=True),
                    gr.skip(), gr.skip(),
                )
                return
            with running_job(request) as cancel:
                yield (
                    gr.update(value="Generating audiobook.", visible=True), gr.skip(), gr.update(interactive=True)
                )
                progress(0, desc="Preparing audiobook")
                try:
                    result = render_model_book(
                        model,
                        session,
                        {**voices, "NARRATOR": _make_voice(ref, transcription, direction)},
                        on_progress=lambda done, total: progress((done, total), desc="Generating audiobook"),
                        cancel_event=cancel,
                        validate_audio=validate,
                        batch_size=batch or PROJECT_BATCH_SIZE_DEFAULT,
                    )
                    cached_result = audiobook.move_resource_to_block_cache(result)
                except ModelRunError as exc:
                    message = str(exc)
                except Exception:
                    message = "Could not generate the audiobook."
                else:
                    yield (
                        gr.update(value="Audiobook ready.", visible=True),
                        gr.update(value=cached_result, visible=True),
                        gr.update(interactive=False),
                    )
                    return
            yield gr.update(value=message, visible=True), gr.skip(), gr.update(interactive=False)

        generate_book.click(
            generate_audiobook,
            [book, cast, book_model, reference, transcript, instructions, validate_audio, batch_size],
            [book_status, audiobook, stop_book],
            **PRIVATE_EVENT,
        )

        def compare_samples(
            models, text, language, ref, transcription, direction, request: gr.Request, progress=gr.Progress(),
        ):
            selected = list(dict.fromkeys(models or []))
            if not selected or any(model not in model_labels for model in selected) or not (text or "").strip():
                yield [
                    gr.update(value="Enter text and select at least one installed speech model.", visible=True),
                    gr.skip(), *[gr.skip() for _ in sample_players],
                ]
                return

            voice = _make_voice(ref, transcription, direction)
            with running_job(request) as cancel:
                yield [
                    gr.update(value="Generating text samples.", visible=True),
                    gr.update(interactive=True),
                    *[gr.update(value=None, visible=False) for _ in sample_players],
                ]
                statuses = []
                for index, model in enumerate(selected):
                    if cancel.is_set():
                        break
                    label = model_labels[model]
                    progress((index, len(selected)), desc=f"Generating with {label}")
                    updates = [gr.skip() for _ in sample_players]
                    try:
                        audio = render_model_sample(
                            model, text, voice, language_code=language or "en", cancel_event=cancel,
                        )
                    except ModelRunError as exc:
                        if cancel.is_set():
                            break
                        statuses.append(f"{label} — {exc}")
                    except Exception:
                        statuses.append(f"{label} — generation failed.")
                    else:
                        statuses.append(f"{label} — ready.")
                        updates[index] = gr.update(label=label, value=audio, visible=True)
                    yield [gr.update(value="\n".join(statuses), visible=True), gr.skip(), *updates]
                if cancel.is_set():
                    statuses.append("Generation stopped.")
            yield [
                gr.update(value="\n".join(statuses), visible=True),
                gr.update(interactive=False),
                *[gr.skip() for _ in sample_players],
            ]

        generate_samples.click(
            compare_samples,
            [sample_models, sample_text, sample_language, reference, transcript, instructions],
            [sample_status, stop_samples, *sample_players],
            **PRIVATE_EVENT,
        )
        stop_options = {"api_visibility": "private", "queue": False, "show_progress": "hidden"}
        stop_book.click(stop_generation, **stop_options)
        stop_samples.click(stop_generation, **stop_options)

    return app.queue(default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local TTS Audiobook Tool browser interface.")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--open", action="store_true", help="Open the browser after launch.")
    args = parser.parse_args()
    build_app().launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=args.open,
        share=False,
        show_error=False,
        footer_links=[],
        max_file_size="100mb",
        enable_monitoring=False,
        ssr_mode=False,
        mcp_server=False,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="sky",
            font=["system-ui", "sans-serif"],
            font_mono=["ui-monospace", "monospace"],
        ),
    )


if __name__ == "__main__":
    main()
