from __future__ import annotations

import argparse
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Lock

import gradio as gr

from tts_audiobook_tool import web_models
from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.constants_config import (
    PROJECT_BATCH_SIZE_DEFAULT, PROJECT_BATCH_SIZE_MAX,
    PROJECT_MAX_RETRIES_DEFAULT, PROJECT_MAX_RETRIES_MAX, PROJECT_MAX_RETRIES_MIN,
)
from tts_audiobook_tool.l import L
from tts_audiobook_tool.web_abr_project import read_abr_project
from tts_audiobook_tool.web_activity_panel import ACTIVITY_CSS, ActivityPanel
from tts_audiobook_tool.web_api_keys import StoredApiKeyError, forget_api_key, has_api_key, load_api_key, save_api_key
from tts_audiobook_tool.web_backend import (
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    UNIDENTIFIED_CAST_ROLES,
    BookSession,
    WebBackendError,
    analyze_book_dialogue,
    book_cast_problem,
    close_book_session,
)
from tts_audiobook_tool.web_book_output_panel import create_book_output_panel
from tts_audiobook_tool.web_character_merge import merge_characters, suggest_duplicate_characters
from tts_audiobook_tool.web_chat_panel import VoiceChatPanel
from tts_audiobook_tool.web_dialogue import (
    correct_speaker, dialogue_rows, dialogue_sections, display_voice_direction, speaker_choices,
)
from tts_audiobook_tool.web_projects import (
    WebProjectError,
    create_project,
    list_projects,
    load_project,
    open_project_folder,
    save_project,
)
from tts_audiobook_tool.web_legacy_project import (
    WebLegacyProjectError, list_console_projects, load_console_project,
)
from tts_audiobook_tool.web_line_voice_panel import LineVoicePanel
from tts_audiobook_tool.web_model_panel import ModelPanel
from tts_audiobook_tool.web_model_setup import ModelSetupProgress, WebModelSetupError, install_model, model_profiles
from tts_audiobook_tool.web_playback_panel import PlaybackPanel
from tts_audiobook_tool.web_models import (
    ModelRunError,
    installed_model_choices,
    render_model_book,
    render_model_sample,
)
from tts_audiobook_tool.web_render import RenderVoice, WebRenderError, character_voice_instruction
from tts_audiobook_tool.web_reader import reader_mount
from tts_audiobook_tool.web_remote_models import REMOTE_MODEL_ID, WebRemoteTtsError
from tts_audiobook_tool.web_remote_panel import RemoteModelPanel
from tts_audiobook_tool.web_runtime_panel import create_runtime_panel
from tts_audiobook_tool.web_runtime_settings import WebRuntimeSettingsError
from tts_audiobook_tool.web_text_panel import SectionTextPanel, TextImportPanel
from tts_audiobook_tool.web_tools_panel import build_tools_panel
from tts_audiobook_tool.web_voice_panel import VoicePanel, bind_transcription, voice_status
from tts_audiobook_tool.web_voices import WebVoiceError, import_voices, load_voice, matching_voice_id


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
""" + ACTIVITY_CSS


def _book_summary(session: BookSession) -> str:
    lines = [session.book.title, session.summary()]
    titles = [section.title for section in session.book.sections if section.title]
    if titles:
        lines.extend(["", *titles])
    if session.warnings:
        lines.extend(["", "Import warnings", *session.warnings])
    return "\n".join(lines)


def _narrator_role(session: BookSession | None) -> str:
    narrator = session.roster.narrator if session is not None and session.roster is not None else None
    return narrator.id if narrator is not None else "NARRATOR"


def _cast_role_name(session: BookSession, role: str) -> str | None:
    if session.roster is None:
        return None
    character = session.roster.get(role)
    return (character.name or character.id) if character is not None else UNIDENTIFIED_CAST_ROLES.get(role)


def _review_components(session: BookSession | None, selected_role: str | None = None) -> tuple[dict, dict]:
    roster = session.roster if session is not None else None
    roster_characters = roster.characters if roster is not None else ()
    rows = [
        [character.name, display_voice_direction(character.voice_direction)] for character in roster_characters
        if not character.is_narrator or character.name.strip().casefold() not in {"", "narrator", "the narrator"}
    ]
    choices = []
    if session is not None and session.book.sections:
        choices.append(("Narrator", _narrator_role(session)))
        choices.extend((character.name, character.id) for character in roster_characters if not character.is_narrator)
        if roster is not None:
            choices.extend((label, role) for role, label in UNIDENTIFIED_CAST_ROLES.items())
    return (
        gr.update(value=rows, visible=bool(rows)),
        gr.update(
            choices=choices, value=selected_role if any(role == selected_role for _, role in choices) else None,
            visible=bool(choices),
        ),
    )


def build_app(*, terminal_log: bool = False) -> gr.Blocks:
    """Build the loopback UI without loading models or opening a connection."""
    if not hasattr(L, "logger"):
        L.init("tts-audiobook-web", level=logging.WARNING)
    local_choices = list(installed_model_choices())
    remote_choice = ("Remote TTS server", REMOTE_MODEL_ID)
    model_choices = [*local_choices, remote_choice]
    model_labels = {model_id: label for label, model_id in model_choices}
    default_model = local_choices[0][1] if local_choices else None
    profile_ids = {profile.model_id for profile in model_profiles()}
    running_jobs: dict[str | None, Event] = {}
    running_jobs_lock = Lock()
    activity_panel = ActivityPanel(terminal_log=terminal_log)

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
        model_settings = gr.State({})
        imported_epub = gr.State(None)
        gr.Markdown(f"# {TITLE}", elem_id="app-heading")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=275, elem_id="voice-settings"):
                voice_panel = VoicePanel()

            with gr.Column(scale=3, min_width=345):
                with gr.Tabs():
                    with gr.Tab("Audiobook"):
                        with gr.Group():
                            text_import = TextImportPanel()
                            summary = gr.Textbox(
                                label="Book summary", interactive=False, lines=4, max_lines=12, visible=False
                            )
                        section_text = SectionTextPanel()

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
                                import_console_voices = gr.Button("Import console voices")
                            with gr.Row():
                                project_name = gr.Textbox(label="New project name", scale=3)
                                save_new = gr.Button("Save new project", interactive=False)
                            save_changes = gr.Button("Save changes", interactive=False)
                            open_folder = gr.Button("Open current project folder", size="sm")
                            with gr.Accordion("Create project from ABR settings", open=False):
                                abr_file = gr.File(label="ABR audiobook", file_types=[".flac", ".m4a", ".m4b"])
                                create_abr = gr.Button("Create project from ABR")
                            project_status = gr.Textbox(label="Project status", interactive=False, visible=False)

                        with gr.Accordion("Character voices", open=False):
                            gr.Markdown(
                                "Analyze the book to find characters and label their dialogue. Book text is sent to the configured LLM."
                            )
                            with gr.Row():
                                api_key = gr.Textbox(
                                    label="LLM API key", type="password", scale=3,
                                    html_attributes=gr.InputHTMLAttributes(autocomplete="off"),
                                )
                                save_key = gr.Button("Save key", size="sm")
                                forget_key = gr.Button("Forget saved key", size="sm")
                            key_status = gr.Markdown()
                            with gr.Row():
                                llm_model = gr.Textbox(value=DEFAULT_OPENAI_MODEL, label="LLM model")
                                effort = gr.Dropdown(
                                    REASONING_EFFORTS, value=DEFAULT_REASONING_EFFORT, label="Reasoning effort",
                                    info="Available levels vary by model.", interactive=True,
                                )
                            with gr.Accordion("LLM endpoint and request settings", open=False):
                                llm_endpoint = gr.Textbox(value=DEFAULT_LLM_ENDPOINT, label="Chat completions URL")
                                llm_extra_params = gr.Textbox(
                                    label="Extra parameters for book analysis (JSON)", lines=3,
                                )
                            character_voices = gr.Checkbox(
                                label="Use subtle Breeze direction for unassigned characters", value=True,
                                info="Use the narrator voice as the starting point in local Breeze TTS 2. "
                                     "Book analysis identifies gender when supported by the text and may suggest a slight "
                                     "change in tone or pace. Your voice assignments take precedence.",
                            )
                            analyze_performance_cues = gr.Checkbox(
                                label="Ask the LLM for dialogue performance cues", value=True,
                                info="Propose delivery directions and vocal events for local Breeze TTS 2.",
                            )
                            with gr.Row():
                                analyze = gr.Button("Analyze book", interactive=False)
                                analysis_status = gr.Textbox(
                                    label="Analysis status", interactive=False, visible=False, scale=3,
                                )
                            characters = gr.Dataframe(
                                label="Identified characters",
                                headers=["Character", "Voice direction"], datatype="str",
                                interactive=False,
                                max_height=340,
                                visible=False,
                            )
                            with gr.Accordion("Merge characters", open=False, visible=False) as merge_panel:
                                duplicate_suggestion = gr.Dropdown(
                                    label="Possible duplicates", choices=[], interactive=True, visible=False,
                                )
                                with gr.Row():
                                    keep_character = gr.Dropdown(
                                        label="Character to keep", choices=[], interactive=True,
                                        info="Keep this character's voice.",
                                    )
                                    remove_character = gr.Dropdown(
                                        label="Character to merge into it", choices=[], interactive=True,
                                    )
                                merge_preview = gr.Textbox(label="Merge summary", interactive=False, visible=False)
                                with gr.Row():
                                    preview_merge = gr.Button("Review merge", size="sm")
                                    confirm_merge = gr.Button("Confirm merge", size="sm", interactive=False)
                                merge_draft = gr.State(None)
                            cast_role = gr.Dropdown(
                                label="Character to cast", choices=[], interactive=True, visible=False
                            )
                            with gr.Group(visible=False) as cast_editor:
                                cast_mode = gr.Radio(
                                    [("Saved voice", "saved"), ("Use narrator voice", "narrator"),
                                     ("Custom voice", "custom"), ("Automatic", "auto")],
                                    value="saved", label="Voice source", interactive=True,
                                )
                                cast_library_voice = gr.Dropdown(label="Choose a saved voice", choices=[], interactive=True)
                                cast_status = gr.Textbox(label="Casting status", interactive=False, visible=False)
                                with gr.Accordion("Character reference and transcript", open=True, visible=False) as cast_details:
                                    cast_reference = gr.Audio(
                                        sources=["upload", "microphone"], type="filepath", editable=False,
                                        label="Character reference audio", buttons=[],
                                    )
                                    cast_transcript = gr.Textbox(
                                        label="Character reference transcript", lines=2, interactive=True,
                                    )
                                    cast_transcription_status = gr.Textbox(
                                        label="Character transcription status", interactive=False, visible=False,
                                    )
                                    transcribe_cast = gr.Button("Retry character transcription", size="sm")
                                    cast_instructions = gr.Textbox(label="Default delivery instructions", lines=2)
                                save_cast = gr.Button("Save assignment")
                                cast_draft = gr.State(RenderVoice())
                                cast_draft_role = gr.State(None)

                            with gr.Accordion("Review dialogue assignments", open=False, visible=False) as review_panel:
                                review_section = gr.Dropdown(label="Book section", choices=[], interactive=True)
                                review_lines = gr.Dataframe(
                                    headers=["Line", "Dialogue", "Speaker", "Uncertain", "Performance cues", "Voice direction"],
                                    datatype=["number", "str", "str", "bool", "str", "str"],
                                    column_widths=[75, "30%", "15%", 85, "20%", "20%"],
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
                            line_voice_panel = LineVoicePanel()
                            cast_language = gr.Textbox(value="en", visible=False)

                        with gr.Group():
                            book_model = gr.Dropdown(
                                choices=model_choices, value=default_model, label="Speech model", interactive=True
                            )
                            with gr.Accordion("Generation settings", open=False):
                                use_performance_cues = gr.Checkbox(
                                    label="Use saved Breeze performance cues", value=True,
                                    info="For local Breeze TTS 2 generation and realtime playback.",
                                )
                                validate_audio = gr.Checkbox(value=True, label="Validate audio with Whisper")
                                with gr.Row():
                                    validation_strictness = gr.Dropdown(
                                        choices=[("Recommended", ""), *((item.label, item.id) for item in Strictness)],
                                        value="", label="Validation tolerance", interactive=True,
                                    )
                                    max_retries = gr.Number(
                                        value=PROJECT_MAX_RETRIES_DEFAULT, precision=0, minimum=PROJECT_MAX_RETRIES_MIN,
                                        maximum=PROJECT_MAX_RETRIES_MAX, label="Maximum retries",
                                    )
                                    batch_size = gr.Number(
                                        value=PROJECT_BATCH_SIZE_DEFAULT, precision=0, minimum=1,
                                        maximum=PROJECT_BATCH_SIZE_MAX, label="Batch size",
                                    )
                                use_recommended_batch = gr.Button("Use recommended batch size", size="sm")
                                auto_concat = gr.Checkbox(value=True, label="Export M4B when all lines are ready")
                                with gr.Row():
                                    first_line = gr.Number(value=1, precision=0, minimum=1, label="First line")
                                    last_line = gr.Number(
                                        value=None, precision=0, minimum=1, label="Last line", info="Blank for end of book.",
                                    )
                                regenerate_lines = gr.Checkbox(value=False, label="Generate new takes for this range")
                            with gr.Row():
                                generate_book = gr.Button(
                                    "Generate or resume audiobook", variant="primary", interactive=False,
                                )
                                stop_book = gr.Button("Stop generation", variant="stop", interactive=False)
                            book_progress = gr.HTML(visible=False)
                            book_status = gr.Textbox(label="Audiobook status", interactive=False, visible=False)
                            audiobook = gr.DownloadButton("Download M4B", visible=False)
                        create_book_output_panel(book, private_event=PRIVATE_EVENT)
                        playback_panel = PlaybackPanel()

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
                                for _ in model_labels.keys() | profile_ids
                            ]

                    with gr.Tab("Speech Models"):
                        model_panel = ModelPanel(local_choices, default_model, model_settings, PRIVATE_EVENT)
                        remote_panel = RemoteModelPanel()
                    with gr.Tab("Voice Chat"):
                        chat_panel = VoiceChatPanel(model_choices, default_model)
                    with gr.Tab("Tools"):
                        build_tools_panel(event_options=PRIVATE_EVENT)
                    with gr.Tab("Settings"):
                        runtime_panel = create_runtime_panel(book, private_event=PRIVATE_EVENT, app=app)

                activity_panel.bind(app, book_progress)

        voice_panel.bind(app, cast_library_voice, event_options=PRIVATE_EVENT)
        text_import.bind(event_options=PRIVATE_EVENT)
        line_voice_panel.bind(app, book, event_options=PRIVATE_EVENT)
        remote_panel.bind(event_options=PRIVATE_EVENT)

        key_saved = "API key saved for this endpoint."
        key_missing = "No API key saved for this endpoint."

        def saved_key_status(endpoint):
            try:
                return key_saved if has_api_key(endpoint or "") else key_missing
            except StoredApiKeyError as exc:
                return str(exc)

        def save_llm_key(endpoint, key):
            try:
                save_api_key(endpoint or "", key or "")
            except StoredApiKeyError as exc:
                return gr.skip(), str(exc)
            return "", key_saved

        def forget_llm_key(endpoint):
            try:
                forget_api_key(endpoint or "")
            except StoredApiKeyError as exc:
                return str(exc)
            return key_missing

        key_events = {**PRIVATE_EVENT, "show_progress": "hidden"}
        app.load(saved_key_status, [llm_endpoint], [key_status], **key_events)
        llm_endpoint.change(saved_key_status, [llm_endpoint], [key_status], **key_events)
        save_key.click(save_llm_key, [llm_endpoint, api_key], [api_key, key_status], **key_events)
        forget_key.click(forget_llm_key, [llm_endpoint], [key_status], **key_events)

        chat_panel.bind(
            api_key=api_key, llm_model=llm_model, effort=effort, endpoint=llm_endpoint,
            voice_panel=voice_panel, model_settings=model_settings, runtime_panel=runtime_panel,
            remote_panel=remote_panel, console_for=activity_panel.console_for,
            event_options=PRIVATE_EVENT,
        )
        playback_panel.bind(
            book, cast, book_model, validate_audio, batch_size, model_settings, validation_strictness, max_retries,
            voice_panel, event_options=PRIVATE_EVENT, runtime_settings=runtime_panel, remote_panel=remote_panel,
            console_for=activity_panel.console_for, performance_cues=use_performance_cues,
            character_voices=character_voices,
        )

        def present_book(session, voices, problem=None):
            table, roles = _review_components(session)
            has_text = session is not None and bool(session.book.sections)
            notice = session.attribution_notice if session else ""
            return (
                session,
                voices,
                gr.update(value=_book_summary(session) if session else problem, visible=bool(session or problem)),
                table,
                roles,
                gr.update(visible=False),
                voice_status(notice or None),
                gr.update(interactive=has_text),
                gr.update(interactive=has_text and bool(model_choices)),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                1, None, False,
            )

        def load_book(previous: BookSession | None, active_id: str | None, *fields):
            try:
                session = text_import.import_book(*fields)
            except WebBackendError as exc:
                problem = str(exc)
            except Exception:
                problem = "Could not import the book text."
            else:
                using_template = previous is not None and not previous.book.sections
                if using_template:
                    session = replace(
                        session, project_dir=previous.project_dir if active_id else session.project_dir,
                        project_language=previous.project_language, word_substitutions=previous.word_substitutions,
                    )
                close_book_session(previous)
                source = fields[1] if fields[0] == "epub" else None
                project_id = active_id if using_template else None
                delivery_default = True if previous is not None and not using_template else gr.skip()
                return (
                    *present_book(session, {}),
                    project_id, gr.update(interactive=True), gr.update(interactive=bool(project_id)),
                    gr.update(value=None, visible=False),
                    source, gr.skip() if source else gr.update(value=None),
                    delivery_default, delivery_default, delivery_default,
                )
            return (
                *[gr.skip()] * (len(import_outputs) - 1),
                gr.update(value=problem, visible=True), *[gr.skip()] * 5,
            )

        book_outputs = [
            book, cast, summary, characters, cast_role, cast_editor, analysis_status,
            analyze, generate_book, book_status, audiobook, first_line, last_line, regenerate_lines,
        ]
        project_actions = [active_project, save_new, save_changes, project_status]
        import_outputs = [*book_outputs, *project_actions]
        imported_book = text_import.load.click(
            load_book, [book, active_project, *text_import.inputs],
            [*import_outputs, imported_epub, text_import.epub,
             use_performance_cues, character_voices, analyze_performance_cues], **PRIVATE_EVENT,
        )

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
                source = text_import.epub.move_resource_to_block_cache(loaded.source_epub)
                narrator_fields = voice_panel.show(loaded.narrator)
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
                source_kind = loaded.session.book.text_source_kind
                if source_kind not in {"epub", "plain_text", "manual"}:
                    source_kind = "epub" if source else "manual"
                return (
                    *present_book(loaded.session, loaded.voices),
                    saved_id, gr.update(interactive=True), gr.update(interactive=saved_id is not None),
                    gr.update(value=message, visible=True),
                    source, source_kind,
                    gr.update(value=source, visible=source_kind == "epub"),
                    gr.update(value=None, visible=source_kind == "plain_text"),
                    gr.update(value="", visible=source_kind == "manual"),
                    *text_import.restore_settings(loaded.session),
                    *narrator_fields, *voice_panel.reset_selection(), voice_status(),
                    loaded.model_id if installed else None,
                    loaded.validate_audio, loaded.batch_size, name,
                    loaded.model_options,
                    loaded.strictness, loaded.max_retries, loaded.auto_concat,
                    getattr(loaded, "performance_cues", False),
                    getattr(loaded, "character_voices", False),
                    getattr(loaded, "analyze_performance_cues", False),
                )
            if loaded is not None:
                close_book_session(loaded.session)
            return (
                *[gr.skip()] * (len(import_outputs) - 1),
                gr.update(value=message, visible=True),
                *[gr.skip()] * (len(opened_project_outputs) - len(import_outputs)),
            )

        def open_console_snapshot(project_id, previous):
            return open_project_snapshot(project_id, previous, from_console=True)

        def import_console_library(project_id, selected, cast_selected):
            loaded = None
            try:
                loaded = load_console_project(project_id or "", model_choices=model_choices)
                summaries = import_voices(loaded.named_voices, upload_root=loaded.session.project_dir)
                updates = voice_panel.refresh(selected, cast_selected)
            except (WebLegacyProjectError, WebVoiceError) as exc:
                return *[gr.skip()] * len(voice_library_outputs), voice_status(str(exc))
            finally:
                if loaded is not None:
                    close_book_session(loaded.session)
            count = len(summaries)
            message = f"Imported {count} console {'voice' if count == 1 else 'voices'}." if count else "No named voices are available to import."
            return *updates, voice_status(message)

        def store_project(
            operation, destination, session, voices, model, source, validate, batch, settings,
            strictness, retries, export, performance_cues, character_voices, analyze_performance_cues, *narrator_fields,
        ):
            try:
                if session is None:
                    raise WebProjectError("Import book text first.")
                saved = operation(
                    destination or "", session, narrator=voice_panel.snapshot(*narrator_fields),
                    voices=voices, model_id=model or "", source_epub=source,
                    validate_audio=validate, batch_size=batch or PROJECT_BATCH_SIZE_DEFAULT,
                    model_options=settings,
                    strictness=strictness, max_retries=retries, auto_concat=export,
                    performance_cues=performance_cues, character_voices=character_voices,
                    analyze_performance_cues=analyze_performance_cues,
                )
            except (WebProjectError, WebRenderError) as exc:
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
            *import_outputs, imported_epub, text_import.source, text_import.epub, text_import.text_file,
            text_import.pasted_text, *text_import.settings_outputs, *voice_panel.outputs, *voice_panel.selection_outputs,
            voice_panel.library_status, book_model,
            validate_audio, batch_size, project_name, model_settings, validation_strictness, max_retries, auto_concat,
            use_performance_cues, character_voices, analyze_performance_cues,
        ]
        opened_project = open_project.click(
            open_project_snapshot, [saved_projects, book], opened_project_outputs, **PRIVATE_EVENT,
        )
        copied_console = copy_console.click(
            open_console_snapshot, [console_projects, book], opened_project_outputs, **PRIVATE_EVENT,
        )

        def create_abr_project(path, name, previous):
            template = None
            try:
                template = read_abr_project(path, abr_file.GRADIO_CACHE)
                saved = create_project(
                    name or "", template.session, narrator=template.narrator, voices={}, model_id=template.model_id,
                    model_options=template.model_options, batch_size=template.batch_size, strictness=template.strictness,
                    max_retries=template.max_retries, auto_concat=template.auto_concat,
                    performance_cues=True, character_voices=True, analyze_performance_cues=True,
                )
            except WebProjectError as exc:
                updates = [gr.skip()] * len(opened_project_outputs)
                updates[import_outputs.index(project_status)] = voice_status(str(exc))
                return *updates, gr.skip(), gr.skip()
            finally:
                if template is not None:
                    close_book_session(template.session)
            updates = list(open_project_snapshot(saved.project_id, previous))
            if isinstance(updates[0], BookSession):
                updates[import_outputs.index(project_status)] = voice_status(
                    f"Created {saved.name} from ABR settings. Import book text and select reference audio."
                )
            return *updates, *refresh_projects(saved.project_id)

        created_from_abr = create_abr.click(
            create_abr_project, [abr_file, project_name, book],
            [*opened_project_outputs, saved_projects, open_project], **PRIVATE_EVENT,
        )

        def show_project_folder(project_id):
            if not project_id:
                return voice_status("Save or open a project first.")
            try:
                folder = open_project_folder(project_id)
                return voice_status(f"Opened {folder}")
            except WebProjectError as exc:
                return voice_status(str(exc))

        open_folder.click(show_project_folder, active_project, project_status, **PRIVATE_EVENT)
        voice_library_outputs = [*voice_panel.selection_outputs, cast_library_voice]
        import_console_voices.click(
            import_console_library, [console_projects, voice_panel.selected, cast_library_voice],
            [*voice_library_outputs, project_status], **PRIVATE_EVENT,
        )
        persist_inputs = [
            book, cast, book_model, imported_epub, validate_audio, batch_size, model_settings,
            validation_strictness, max_retries, auto_concat,
            use_performance_cues, character_voices, analyze_performance_cues, *voice_panel.inputs,
        ]
        persist_outputs = [book, cast, active_project, saved_projects, open_project, save_changes, project_status]
        save_new.click(save_named_project, [project_name, *persist_inputs], persist_outputs, **PRIVATE_EVENT)
        save_changes.click(save_loaded_project, [active_project, *persist_inputs], persist_outputs, **PRIVATE_EVENT)

        def analyze_book(session, key, model, reasoning, endpoint, settings_json, selected_role, character_voices,
                         performance_cues, request: gr.Request):
            console = activity_panel.console_for(request.session_hash)
            secret = (key or "").strip()

            def redact(message: str) -> str:
                return message.replace(secret, "[redacted]") if secret else message

            def log(message: str) -> None:
                console(redact(message).rstrip() + "\n")

            console("\nAnalyzing book.\n")
            yield gr.skip(), gr.skip(), gr.skip(), voice_status("Analyzing characters and dialogue…")
            try:
                if session is None:
                    raise WebBackendError("Import book text first.")
                if not secret:
                    secret = load_api_key(endpoint or "")
                if secret and secret in (endpoint or ""):
                    raise WebBackendError("Remove the API key from the chat completions URL.")
                session = analyze_book_dialogue(
                    session, secret, model, effort=reasoning or "", endpoint=endpoint,
                    settings_json=settings_json or "", on_log=log, performance_cues=performance_cues,
                    character_voices=character_voices,
                )
            except (StoredApiKeyError, WebBackendError) as exc:
                message = str(exc)
            except Exception:
                message = "Book analysis failed. Check your LLM settings and try again."
            else:
                notice = redact(session.attribution_notice)
                if notice != session.attribution_notice:
                    session = replace(session, attribution_notice=notice)
                message = notice or redact("Book analysis complete.")
                yield session, *_review_components(session, selected_role), voice_status(message)
                return
            log(message)
            yield gr.skip(), gr.skip(), gr.skip(), voice_status(redact(message))

        analyzed = analyze.click(
            analyze_book, [
                book, api_key, llm_model, effort, llm_endpoint, llm_extra_params, cast_role, character_voices,
                analyze_performance_cues,
            ],
            [book, characters, cast_role, analysis_status], show_progress="hidden", **PRIVATE_EVENT,
        )

        cast_fields = [
            cast_mode, cast_library_voice, cast_details, cast_reference, cast_transcript,
            cast_instructions, cast_draft, cast_draft_role, cast_transcription_status, cast_status,
        ]
        cast_inputs = [book, cast, cast_role]
        cast_outputs = [cast, *cast_fields]
        narrator_cast_outputs = [*voice_panel.outputs, *voice_panel.selection_outputs, voice_panel.library_status]

        def cast_fields_for(role, voice, message, *, mode="saved", selected=None, is_narrator=False):
            return (
                gr.update(value=mode, visible=not is_narrator),
                gr.update(value=selected, visible=mode == "saved"), gr.update(visible=mode == "custom"),
                cast_reference.move_resource_to_block_cache(voice.reference_path),
                gr.update(value=voice.transcript, interactive=True), voice.instructions,
                voice, role, voice_status(), voice_status(message),
            )

        def unassigned_cast_message(session, role, role_name, use_character_voices):
            character = session.roster.get(role) if session.roster is not None else None
            if use_character_voices and character is not None and character_voice_instruction(character):
                return f"{role_name} uses the narrator voice with subtle character direction in local Breeze TTS 2."
            return f"{role_name} uses the narrator voice by default."

        def show_cast_voice(role, voices, session, narrator_selected, narrator_name, use_character_voices):
            if role is None or session is None:
                return gr.update(visible=False), *cast_fields_for(None, RenderVoice(), None)
            if role == _narrator_role(session):
                message = f"Narrator uses {narrator_name}." if narrator_name else "Narrator uses the voice in Voice settings."
                return gr.update(visible=True), *cast_fields_for(
                    role, RenderVoice(), message, selected=narrator_selected, is_narrator=True,
                )
            role_name = _cast_role_name(session, role)
            if role_name is None:
                return gr.update(visible=False), *cast_fields_for(None, RenderVoice(), None)
            voice = voices.get(role)
            if voice is None:
                fields = cast_fields_for(
                    role, RenderVoice(), unassigned_cast_message(session, role, role_name, use_character_voices),
                )
            elif voice == RenderVoice():
                fields = cast_fields_for(role, voice, f"{role_name} uses the narrator voice.", mode="narrator")
            else:
                selected = matching_voice_id(voice)
                if selected is not None:
                    message = f"{role_name} uses {voice.name}."
                elif voice.name:
                    message = f"{role_name} uses a saved copy of {voice.name}."
                else:
                    message = f"{role_name} uses a custom voice."
                try:
                    fields = cast_fields_for(
                        role, voice, message, mode="saved" if voice.name else "custom", selected=selected,
                    )
                except OSError:
                    fields = cast_fields_for(role, RenderVoice(), f"Could not open the voice assigned to {role_name}.")
            return gr.update(visible=True), *fields

        cast_role.change(
            show_cast_voice, [cast_role, cast, book, voice_panel.selected, voice_panel.name, character_voices],
            [cast_editor, *cast_fields], **PRIVATE_EVENT,
        )

        def refresh_unassigned_cast_status(session, voices, role, use_character_voices):
            if session is None or role is None or role in voices or role == _narrator_role(session):
                return gr.skip()
            role_name = _cast_role_name(session, role)
            if role_name is None:
                return gr.skip()
            return voice_status(unassigned_cast_message(session, role, role_name, use_character_voices))

        character_voices.input(
            refresh_unassigned_cast_status, [*cast_inputs, character_voices], cast_status,
            show_progress="hidden", **PRIVATE_EVENT,
        )
        cast_mode.input(
            lambda mode: (gr.update(visible=mode == "saved"), gr.update(visible=mode == "custom")),
            cast_mode, [cast_library_voice, cast_details], show_progress="hidden", **PRIVATE_EVENT,
        )

        def save_cast_assignment(session, voices, role, draft_role, mode, selected, draft, reference, transcript, instructions):
            narrator_unchanged = tuple(gr.skip() for _ in narrator_cast_outputs)
            try:
                if session is None or role is None or role != draft_role:
                    raise WebVoiceError("Choose a character and wait for its voice settings to load.")
                is_narrator = role == _narrator_role(session)
                if is_narrator or mode == "saved":
                    if not selected:
                        raise WebVoiceError("Choose a saved voice before saving the assignment.")
                    voice = load_voice(selected).voice
                elif mode in {"auto", "narrator"}:
                    voice = RenderVoice()
                elif mode == "custom":
                    voice = replace(
                        draft, reference_path=reference, transcript=(transcript or "").strip(),
                        instructions=(instructions or "").strip(), trim_start=draft.trim_start if reference else 0.0,
                        trim_end=draft.trim_end if reference else None,
                    )
                    if voice == RenderVoice():
                        raise WebVoiceError("Add a custom reference or choose a different voice source.")
                else:
                    raise WebVoiceError("Choose a voice source before saving the assignment.")
                if is_narrator:
                    narrator_fields = voice_panel.show(voice)
                    *selection_updates, _ = voice_panel.refresh(selected, selected)
                    fields = cast_fields_for(
                        role, RenderVoice(), f"Saved {voice.name} for the narrator.", selected=selected, is_narrator=True,
                    )
                    return (
                        gr.skip(), *fields, *narrator_fields, *selection_updates, voice_status(),
                    )
                role_name = _cast_role_name(session, role)
                if role_name is None:
                    raise WebVoiceError("Choose a role from the casting list.")
                if mode == "auto":
                    updated_voices = dict(voices)
                    updated_voices.pop(role, None)
                    fields = cast_fields_for(role, voice, f"Restored automatic casting for {role_name}.", mode=mode)
                    return updated_voices, *fields, *narrator_unchanged
                if mode == "narrator":
                    message = f"Saved the narrator voice for {role_name}."
                else:
                    message = f"Saved {voice.name or 'a custom voice'} for {role_name}."
                fields = cast_fields_for(role, voice, message, mode=mode, selected=selected if mode == "saved" else None)
                return {**voices, role: voice}, *fields, *narrator_unchanged
            except (WebVoiceError, WebRenderError) as exc:
                message = str(exc)
            except OSError:
                message = "Could not open the saved reference."
            return *[gr.skip()] * (len(cast_outputs) - 1), voice_status(message), *narrator_unchanged

        save_cast.click(
            save_cast_assignment,
            [*cast_inputs, cast_draft_role, cast_mode, cast_library_voice, cast_draft,
             cast_reference, cast_transcript, cast_instructions],
            [*cast_outputs, *narrator_cast_outputs], **PRIVATE_EVENT,
        )
        bind_transcription(
            cast_reference, cast_language, cast_transcript, cast_transcription_status, transcribe_cast,
            event_options=PRIVATE_EVENT, draft=cast_draft, reset=((cast_draft, RenderVoice()),),
        )
        book.change(
            lambda session: session.language_code if session else "en", book, cast_language,
            show_progress="hidden", **PRIVATE_EVENT,
        )
        cast_reference.clear(
            lambda: (gr.update(value="", interactive=True), voice_status()),
            outputs=[cast_transcript, cast_transcription_status], trigger_mode="multiple", **PRIVATE_EVENT,
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
                    value=[
                        [row.first_line + 1, row.text, row.speaker, row.uncertain, row.performance, row.voice_direction]
                        for row in rows
                    ],
                    visible=bool(rows),
                ),
                gr.update(choices=choices, value=selected_line, visible=bool(rows)),
                gr.update(choices=speaker_choices(session) if rows else [], value=None, visible=bool(rows)),
                gr.update(visible=bool(rows)),
            )

        def display_review_section(session, section):
            return (*section_controls(session, section), gr.update(value=None, visible=False))

        def update_dialogue_speaker(session, section, line, speaker):
            try:
                if session is None:
                    raise WebBackendError("Import book text first.")
                session = correct_speaker(session, section, line, speaker)
            except WebBackendError as exc:
                return (*[gr.skip()] * 5, gr.update(value=str(exc), visible=True))
            return (
                session, *section_controls(session, section, line),
                gr.update(value="Speaker assignment updated.", visible=True),
            )

        def merge_character_labels(session):
            roster = session.roster if session else None
            if roster is None:
                return {}
            names = {character.id: character.name.strip() or character.id for character in roster.characters}
            counts = Counter(name.casefold() for name in names.values())
            return {
                role: f"{name} ({role})" if counts[name.casefold()] > 1 else name
                for role, name in names.items()
            }

        def reset_merge_preview():
            return None, voice_status(), gr.update(interactive=False)

        def refresh_character_merge(session):
            labels = merge_character_labels(session)
            keep_choices = [(label, role) for role, label in labels.items()]
            remove_choices = [(label, role) for label, role in keep_choices if role != _narrator_role(session)]
            suggestions = [
                (f"{labels[first]} / {labels[second]}", f"{first}/{second}")
                for first, second in suggest_duplicate_characters(session)
            ]
            label = "Merge characters"
            if suggestions:
                plural = "" if len(suggestions) == 1 else "s"
                label += f" ({len(suggestions):,} possible duplicate{plural})"
            return (
                gr.update(label=label, visible=len(labels) > 1 and bool(remove_choices)),
                gr.update(choices=suggestions, value=None, visible=bool(suggestions)),
                gr.update(choices=keep_choices, value=None), gr.update(choices=remove_choices, value=None),
                *reset_merge_preview(),
            )

        def select_merge_suggestion(pair, session):
            if not pair:
                return None, None, *reset_merge_preview()
            keep, remove = pair.split("/", 1)
            if remove == _narrator_role(session):
                keep, remove = remove, keep
            return keep, remove, *reset_merge_preview()

        def merge_selection(session, voices, keep, remove, narrator_name):
            return id(session), keep, remove, voices.get(keep), narrator_name

        def draft_character_merge(session, voices, keep, remove, narrator_name):
            labels = merge_character_labels(session)
            problem = None
            if keep not in labels or remove not in labels or keep == remove:
                problem = "Choose two different characters to merge."
            elif remove == _narrator_role(session):
                problem = "Choose the narrator as the character to keep."
            if problem:
                return None, voice_status(problem), gr.update(interactive=False)
            voice = voices.get(keep)
            if keep == _narrator_role(session) or voice == RenderVoice():
                voice_label = narrator_name or "the narrator voice"
            elif voice is None:
                voice_label = "chosen automatically"
            else:
                voice_label = voice.name or "the retained character's custom voice"
            summary = (
                f"Merge {labels[remove]} into {labels[keep]}? "
                f"The merged character's voice will be {voice_label}."
            )
            return (
                merge_selection(session, voices, keep, remove, narrator_name), voice_status(summary),
                gr.update(interactive=True),
            )

        merge_preview_outputs = [merge_draft, merge_preview, confirm_merge]
        merge_inputs = [book, cast, keep_character, remove_character, voice_panel.name]
        book.change(
            refresh_character_merge, book,
            [merge_panel, duplicate_suggestion, keep_character, remove_character, *merge_preview_outputs],
            show_progress="hidden", **PRIVATE_EVENT,
        )
        duplicate_suggestion.input(
            select_merge_suggestion, [duplicate_suggestion, book],
            [keep_character, remove_character, *merge_preview_outputs], show_progress="hidden", **PRIVATE_EVENT,
        )
        gr.on(
            [keep_character.input, remove_character.input], reset_merge_preview, outputs=merge_preview_outputs,
            show_progress="hidden", **PRIVATE_EVENT,
        )
        preview_merge.click(
            draft_character_merge, merge_inputs, merge_preview_outputs, show_progress="hidden", **PRIVATE_EVENT,
        )

        def confirm_character_merge(session, voices, keep, remove, narrator_name, draft, role, section, line):
            try:
                if draft != merge_selection(session, voices, keep, remove, narrator_name):
                    raise WebBackendError("Review the current selection before confirming the merge.")
                labels = merge_character_labels(session)
                session, voices = merge_characters(session, voices, keep, remove)
            except WebBackendError as exc:
                return (
                    *[gr.skip()] * (len(merge_outputs) - len(merge_preview_outputs)),
                    None, voice_status(str(exc)), gr.update(interactive=False),
                )
            message = f"Merged {labels[remove]} into {labels[keep]}. Save the project to keep this change."
            return (
                session, voices, *_review_components(session, keep if role == remove else role),
                voice_status(message), *section_controls(session, section, line), *reset_merge_preview(),
            )

        merge_outputs = [
            book, cast, characters, cast_role, analysis_status,
            review_lines, review_line, review_speaker, save_speaker, *merge_preview_outputs,
        ]
        confirm_merge.click(
            confirm_character_merge, [*merge_inputs, merge_draft, cast_role, review_section, review_line],
            merge_outputs, show_progress="hidden", **PRIVATE_EVENT,
        )

        for event in (imported_book, opened_project, copied_console, created_from_abr):
            event.then(
                section_text.refresh, book, section_text.refresh_outputs,
                show_progress="hidden", **PRIVATE_EVENT,
            )
            event.then(
                update_review_sections, [book], [review_panel, review_section],
                show_progress="hidden", **PRIVATE_EVENT,
            )
        section_text.bind(
            book, book_changed=lambda session, voices: (*present_book(session, voices), *update_review_sections(session, None)),
            changed_inputs=[cast], changed_outputs=[*book_outputs, review_panel, review_section],
            event_options=PRIVATE_EVENT, source_epub=imported_epub,
        )
        review_outputs = [review_lines, review_line, review_speaker, save_speaker, review_status]
        analyzed.then(
            update_review_sections, [book, review_section], [review_panel, review_section],
            show_progress="hidden", **PRIVATE_EVENT,
        ).then(
            display_review_section, [book, review_section], review_outputs,
            show_progress="hidden", **PRIVATE_EVENT,
        )
        review_section.change(display_review_section, [book, review_section], review_outputs, **PRIVATE_EVENT)
        save_speaker.click(
            update_dialogue_speaker, [book, review_section, review_line, review_speaker], [book, *review_outputs],
            **PRIVATE_EVENT,
        )

        def generate_audiobook(
            session, voices, model, validate, batch, settings,
            strictness, retries, export, first, last, regenerate, performance_cues, character_voices,
            force_cpu, whisper_model, whisper_device, save_generation_logs, aac_bitrate, checkpoint_format,
            remote_endpoint, remote_config,
            request: gr.Request, *narrator_fields,
        ):
            if session is None or model not in model_labels:
                yield (
                    gr.update(value="Import book text and choose an installed speech model.", visible=True),
                    gr.skip(), gr.skip(),
                )
                return
            if problem := book_cast_problem(session, first_line=first, last_line=last):
                yield voice_status(problem), gr.skip(), gr.skip()
                return
            try:
                narrator = voice_panel.snapshot(*narrator_fields)
                runtime = runtime_panel.snapshot(
                    force_cpu, whisper_model, whisper_device, save_generation_logs, aac_bitrate, checkpoint_format,
                )
                remote = remote_panel.selection_for(remote_endpoint, remote_config) if model == REMOTE_MODEL_ID else None
            except (WebRenderError, WebRuntimeSettingsError, WebRemoteTtsError) as exc:
                yield voice_status(str(exc)), gr.skip(), gr.skip()
                return
            with running_job(request) as cancel:
                activity = activity_panel.session(request.session_hash)
                activity.start_book()
                activity.write(f"\nGenerating audiobook with {model_labels[model]}\n")
                yield (
                    gr.update(value="Generating audiobook.", visible=True), gr.skip(), gr.update(interactive=True)
                )
                try:
                    result = render_model_book(
                        model,
                        session,
                        {**voices, "NARRATOR": narrator},
                        on_telemetry=activity.update_book,
                        on_console=activity.write,
                        cancel_event=cancel,
                        validate_audio=validate,
                        batch_size=batch or PROJECT_BATCH_SIZE_DEFAULT,
                        model_options=settings.get(model),
                        strictness=strictness or None, max_retries=retries, auto_concat=export,
                        first_line=first, last_line=last, regenerate=regenerate,
                        performance_cues=performance_cues, character_voices=character_voices,
                        runtime_settings=runtime,
                        **({"remote_selection": remote} if remote is not None else {}),
                    )
                    cached_result = audiobook.move_resource_to_block_cache(result) if result else None
                except ModelRunError as exc:
                    message = str(exc)
                except Exception:
                    message = "Could not generate the audiobook."
                else:
                    message = "Audiobook ready." if cached_result else "Selected lines are ready to review or export."
                    activity.finish_book(message)
                    activity.write(message + "\n")
                    yield (
                        gr.update(value=message, visible=True),
                        gr.update(value=cached_result, visible=bool(cached_result)),
                        gr.update(interactive=False),
                    )
                    return
            activity.finish_book("Generation stopped" if cancel.is_set() else "Generation failed")
            activity.write(message + "\n")
            yield gr.update(value=message, visible=True), gr.skip(), gr.update(interactive=False)

        generate_book.click(
            generate_audiobook,
            [
                book, cast, book_model, validate_audio, batch_size, model_settings,
                validation_strictness, max_retries, auto_concat, first_line, last_line, regenerate_lines,
                use_performance_cues, character_voices,
                *runtime_panel.inputs, *remote_panel.selection_inputs, *voice_panel.inputs,
            ],
            [book_status, audiobook, stop_book],
            show_progress="hidden",
            **PRIVATE_EVENT,
        )

        def compare_samples(
            models, text, language, settings, force_cpu, whisper_model, whisper_device, save_generation_logs,
            aac_bitrate, checkpoint_format,
            remote_endpoint, remote_config,
            request: gr.Request, progress=gr.Progress(), *narrator_fields,
        ):
            selected = list(dict.fromkeys(models or []))
            if not selected or any(model not in model_labels for model in selected) or not (text or "").strip():
                yield [
                    gr.update(value="Enter text and select at least one installed speech model.", visible=True),
                    gr.skip(), *[gr.skip() for _ in sample_players],
                ]
                return

            try:
                voice = voice_panel.snapshot(*narrator_fields)
                runtime = runtime_panel.snapshot(
                    force_cpu, whisper_model, whisper_device, save_generation_logs, aac_bitrate, checkpoint_format,
                )
            except (WebRenderError, WebRuntimeSettingsError) as exc:
                yield [voice_status(str(exc)), gr.skip(), *[gr.skip() for _ in sample_players]]
                return
            player_props: list[dict[str, str | bool | None]] = [
                {"value": None, "visible": False} for _ in sample_players
            ]
            with running_job(request) as cancel:
                console = activity_panel.console_for(request.session_hash)
                yield [
                    gr.update(value="Generating text samples.", visible=True),
                    gr.update(interactive=True),
                    *[gr.update(**props) for props in player_props],
                ]
                statuses = []
                for index, model in enumerate(selected):
                    if cancel.is_set():
                        break
                    label = model_labels[model]
                    console(f"\nGenerating text sample with {label}\n")
                    progress((index, len(selected)), desc=f"Generating with {label}")
                    updates = [gr.skip() for _ in sample_players]
                    try:
                        remote = remote_panel.selection_for(
                            remote_endpoint, remote_config,
                        ) if model == REMOTE_MODEL_ID else None
                        audio = render_model_sample(
                            model, text, voice, language_code=language or "en", cancel_event=cancel,
                            model_options=settings.get(model),
                            runtime_settings=runtime,
                            on_console=console,
                            **({"remote_selection": remote} if remote is not None else {}),
                        )
                    except (ModelRunError, WebRemoteTtsError) as exc:
                        if cancel.is_set():
                            break
                        statuses.append(f"{label} — {exc}")
                    except Exception:
                        statuses.append(f"{label} — generation failed.")
                    else:
                        statuses.append(f"{label} — ready.")
                        player_props[index] = {"label": label, "value": audio, "visible": True}
                        updates[index] = gr.update(**player_props[index])
                    console(statuses[-1] + "\n")
                    yield [gr.update(value="\n".join(statuses), visible=True), gr.skip(), *updates]
                if cancel.is_set():
                    statuses.append("Generation stopped.")
                    console("Generation stopped.\n")
            # Gradio uses the last yield as the completed queue response.
            yield [
                gr.update(value="\n".join(statuses), visible=True),
                gr.update(interactive=False),
                *[gr.update(**props) for props in player_props],
            ]

        generate_samples.click(
            compare_samples,
            [
                sample_models, sample_text, sample_language, model_settings, *runtime_panel.inputs,
                *remote_panel.selection_inputs, *voice_panel.inputs,
            ],
            [sample_status, stop_samples, *sample_players],
            show_progress_on=sample_status,
            **PRIVATE_EVENT,
        )
        stop_options = {"api_visibility": "private", "queue": False, "show_progress": "hidden"}
        stop_book.click(stop_generation, **stop_options)
        stop_samples.click(stop_generation, **stop_options)
        model_panel.cancel.click(stop_generation, **stop_options)

        def show_model(model_id):
            detail, available = model_panel.describe(model_id, model_labels)
            return detail, gr.update(interactive=available), gr.update(value=None, visible=False)

        model_panel.selected.input(
            show_model, [model_panel.selected], [model_panel.details, model_panel.install, model_panel.setup_status],
            **PRIVATE_EVENT,
        )
        force_cpu_option = runtime_panel.inputs[0]
        batch_inputs = [book_model, validate_audio, model_settings, force_cpu_option]
        for event in (app.load, book_model.input, use_recommended_batch.click):
            event(model_panel.recommended_batch, batch_inputs, batch_size, show_progress="hidden", **PRIVATE_EVENT)

        def install_selected_model(model_id, sample_selection, session, validate, settings, force_cpu,
                                   request: gr.Request, progress=gr.Progress()):
            unchanged = [gr.skip()] * 8
            console = activity_panel.console_for(request.session_hash)

            def report_setup(update: ModelSetupProgress) -> None:
                progress((update.completed, update.total), desc=update.description)
                console(update.description + "\n")

            with running_job(request) as cancel:
                console("\nInstalling speech model\n")
                yield (
                    *unchanged, gr.update(value="Installing model.", visible=True),
                    gr.update(interactive=False), gr.update(interactive=True), gr.skip(),
                )
                try:
                    install_model(model_id, on_progress=report_setup, cancel_event=cancel)
                    web_models.installed_model_choices.cache_clear()
                    local_choices[:] = installed_model_choices()
                    model_choices[:] = [*local_choices, remote_choice]
                    model_labels.clear()
                    model_labels.update((name, label) for label, name in model_choices)
                    if model_id not in model_labels:
                        raise WebModelSetupError("Setup finished, but the launcher could not find the model environment.")
                except WebModelSetupError as exc:
                    message = str(exc)
                except Exception:
                    message = "Could not install the model."
                else:
                    detail, available = model_panel.describe(model_id, model_labels)
                    samples = dict.fromkeys([*(sample_selection or []), model_id])
                    console(f"{model_labels[model_id]} is installed.\n")
                    yield (
                        gr.update(choices=model_panel.choices(local_choices), value=model_id), detail,
                        gr.update(choices=model_choices, value=model_id),
                        gr.update(choices=model_choices, value=[item for item in samples if item in model_labels]),
                        model_panel.recommended_batch(model_id, validate, settings, force_cpu),
                        gr.update(interactive=session is not None and bool(session.book.sections)),
                        gr.update(interactive=True),
                        gr.update(value=None, visible=False),
                        gr.update(value=f"{model_labels[model_id]} is installed and selected.", visible=True),
                        gr.update(interactive=available), gr.update(interactive=False),
                        gr.update(choices=chat_panel.model_choices(model_choices), value=model_id),
                    )
                    return
            console(message + "\n")
            _, available = model_panel.describe(model_id, model_labels)
            yield (
                *unchanged, gr.update(value=message, visible=True),
                gr.update(interactive=available), gr.update(interactive=False), gr.skip(),
            )

        model_panel.install.click(
            install_selected_model,
            [model_panel.selected, sample_models, book, validate_audio, model_settings, force_cpu_option],
            [
                model_panel.selected, model_panel.details, book_model, sample_models,
                batch_size, generate_book, generate_samples, sample_status,
                model_panel.setup_status, model_panel.install, model_panel.cancel, chat_panel.speech_model,
            ],
            show_progress_on=model_panel.setup_status,
            **PRIVATE_EVENT,
        )

    return app.queue(default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local TTS Audiobook Tool browser interface.")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--open", action="store_true", help="Open the browser after launch.")
    parser.add_argument("--terminal-log", action="store_true", help="Also show browser console output in this terminal.")
    args = parser.parse_args()
    build_app(terminal_log=args.terminal_log).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=args.open,
        share=False,
        show_error=False,
        footer_links=[],
        max_file_size="2gb",
        enable_monitoring=False,
        ssr_mode=False,
        mcp_server=False,
        app_kwargs={"routes": [reader_mount()]},
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
