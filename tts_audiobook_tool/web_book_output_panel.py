"""Browser controls for reviewing and exporting saved audiobook takes."""
from __future__ import annotations

from datetime import datetime

import gradio as gr

from tts_audiobook_tool.app_types import (
    ExportType, HighShelfEq, MinSegmentSentenceSilence, NormalizationType, RoomReverbStereo, RoomTone,
)
from tts_audiobook_tool.constants import AAC_BITRATE_DEFAULT, AAC_BITRATES
from tts_audiobook_tool.constants_config import (
    PROJECT_DEFAULT_BREAK_EFFECT, PROJECT_DEFAULT_LIMIT_SILENCE_GAPS, PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
    SILENCE_GAP_DURATION_MAX, SILENCE_GAP_DURATION_MIN,
)
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.web_book_outputs import (
    BookExportOptions, WebBookOutputError, export_book_audio_parts, get_book_export_path, get_book_take_path,
    list_book_exports, list_book_render_runs, review_book_line,
)
from tts_audiobook_tool.web_reader import reader_link


def create_book_output_panel(book: gr.State, *, private_event: dict) -> None:
    with gr.Accordion("Review and export generated audio", open=False):
        with gr.Row():
            runs = gr.Dropdown(label="Rendering", choices=[], interactive=True, scale=5)
            refresh = gr.Button("Find renderings", scale=1)
        with gr.Row():
            line = gr.Number(label="Line number", value=1, minimum=1, precision=0)
            review = gr.Button("Review line")
        line_text = gr.Textbox(label="Text from this rendering", interactive=False, visible=False)
        takes = gr.Dropdown(label="Available takes", choices=[], interactive=True, visible=False)
        take_audio = gr.Audio(label="Selected take", interactive=False, visible=False)

        with gr.Row():
            first = gr.Number(label="First line", value=1, minimum=1, precision=0)
            last = gr.Number(label="Last line", minimum=1, precision=0, info="Leave blank for the end of the book.")
            file_type = gr.Dropdown(
                label="File type", choices=[(item.label, item.id) for item in ExportType],
                value=ExportType.AAC.id, interactive=True,
            )
        with gr.Accordion("Export options", open=False):
            with gr.Row():
                bookmarks = gr.Textbox(label="Bookmark lines", info="Enter comma-separated line numbers for a single file.")
                cuts = gr.Textbox(label="Split before lines", placeholder="8, 31, 74", info="Each line starts a new file.")
            split_sections = gr.Checkbox(label="Split at saved section boundaries and file split points", value=False)
            with gr.Row():
                bitrate = gr.Dropdown(label="AAC bitrate", choices=AAC_BITRATES, value=AAC_BITRATE_DEFAULT)
                loudness = gr.Dropdown(
                    label="Loudness", choices=[(item.value.label, item.value.id) for item in NormalizationType],
                    value=list(NormalizationType)[0].value.id,
                )
                pauses = gr.Dropdown(
                    label="Pauses between segments", choices=[(item.value.label, item.value.id) for item in ReasonPauseTypes],
                    value=ReasonPauseTypes.default().value.id,
                )
            with gr.Row():
                treble = gr.Dropdown(
                    label="Treble EQ", choices=[(item.id.capitalize(), item.id) for item in HighShelfEq],
                    value=HighShelfEq.DISABLED.id,
                )
                room_tone = gr.Dropdown(
                    label="Room tone", choices=[(item.label, item.id) for item in RoomTone], value=RoomTone.get_default().id,
                )
                stereo = gr.Dropdown(
                    label="Stereo reverb", choices=[(item.label, item.id) for item in RoomReverbStereo],
                    value=RoomReverbStereo.get_default().id,
                )
            with gr.Row():
                section_effect = gr.Checkbox(label="Section break sound effect", value=PROJECT_DEFAULT_BREAK_EFFECT)
                subdivisions = gr.Checkbox(label="Reader uses phrase subdivisions", value=False)
                sentence_pause = gr.Checkbox(label="Minimum pause at sentence boundaries", value=False)
                upsample = gr.Checkbox(label="Generative upsampling with LavaSR v2", value=False)
            with gr.Row():
                limit_gaps = gr.Checkbox(label="Limit long silences", value=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS)
                max_gap = gr.Number(
                    label="Maximum silence in seconds", value=PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
                    minimum=SILENCE_GAP_DURATION_MIN, maximum=SILENCE_GAP_DURATION_MAX,
                )
        with gr.Row():
            export = gr.Button("Export line range", variant="primary")
            export_history = gr.Dropdown(label="Saved exports", choices=[], interactive=True, scale=4)
        status = gr.Textbox(label="Review and export status", interactive=False, visible=False)
        download = gr.File(label="Download export", interactive=False, visible=False)
        reader = gr.HTML(visible=False)

    def show_status(message: str):
        return gr.update(value=message, visible=bool(message))

    def find_renderings(session, current):
        if session is None:
            return gr.update(choices=[], value=None), show_status("Open a project with generated audio.")
        try:
            found = list_book_render_runs(session)
        except OSError:
            return gr.update(choices=[], value=None), show_status("Could not read the project’s rendering folder.")
        choices = [
            (
                f"{run.model_label} · {run.generated_lines:,}/{run.total_lines:,} lines · "
                f"{datetime.fromtimestamp(run.updated_at):%Y-%m-%d %H:%M:%S}" + (f" · {run.title}" if run.title else ""),
                run.run_id,
            )
            for run in found
        ]
        selected = current if any(item.run_id == current for item in found) else (found[0].run_id if found else None)
        message = "" if found else "Generate audio for this project to review and export it here."
        return gr.update(choices=choices, value=selected), show_status(message)

    def select_rendering(session, run_id):
        try:
            choices = list_book_exports(session, run_id) if session is not None and run_id else []
        except (OSError, WebBookOutputError):
            choices = []
        return (
            gr.update(value=None, visible=False), gr.update(choices=[], value=None, visible=False),
            gr.update(value=None, visible=False), gr.update(choices=list(choices), value=None),
            gr.update(value=None, visible=False), gr.update(value=None, visible=False),
        )

    def read_line(session, run_id, line_number):
        try:
            if session is None or not run_id:
                raise WebBookOutputError("Choose a rendering first.")
            reviewed = review_book_line(session, run_id, line_number)
            choices = []
            for number, take in enumerate(reviewed.takes, 1):
                label = "Preferred take" if take.preferred else f"Take {number}"
                if take.word_errors is None:
                    label += " · Unvalidated"
                elif take.word_errors:
                    label += f" · Word errors ({take.word_errors})"
                else:
                    label += " · No word errors"
                if take.failed:
                    label += " · Outside tolerance"
                choices.append((label, take.filename))
            selected = choices[0][1] if choices else None
            message = "" if choices else "No take is available for this line."
            return gr.update(value=reviewed.text, visible=True), gr.update(
                choices=choices, value=selected, visible=bool(choices),
            ), show_status(message)
        except (OSError, WebBookOutputError) as error:
            return gr.update(value=None, visible=False), gr.update(choices=[], value=None, visible=False), show_status(str(error))

    def play_take(session, run_id, line_number, filename):
        if session is None or not run_id or not filename:
            return gr.update(value=None, visible=False), gr.skip()
        try:
            path = get_book_take_path(session, run_id, line_number, filename)
            return gr.update(value=path, visible=True), show_status("")
        except (OSError, WebBookOutputError) as error:
            return gr.update(value=None, visible=False), show_status(str(error))

    def open_export(session, run_id, export_id):
        if session is None or not run_id or not export_id:
            return gr.skip(), gr.skip(), gr.skip()
        try:
            path = get_book_export_path(session, run_id, export_id)
            cached = download.move_resource_to_block_cache(path)
            return gr.update(value=cached, visible=True), gr.update(value=reader_link(cached), visible=True), show_status("")
        except (OSError, WebBookOutputError) as error:
            return gr.update(value=None, visible=False), gr.update(value=None, visible=False), show_status(str(error))

    def make_export(
        session, run_id, first_line, last_line, file_type_id, marks, split_lines, use_sections, bit_rate, loudness_id, pause_id, treble_id,
        tone_id, stereo_id, effect, subdivide, sentence, use_upsampler, trim, gap,
    ):
        try:
            if session is None or not run_id:
                raise WebBookOutputError("Choose a rendering first.")
            container = ExportType.get_by_id(file_type_id)
            if container is None:
                raise WebBookOutputError("Choose a file type.")
            results = export_book_audio_parts(
                session, run_id, first_line=first_line, last_line=last_line, bookmarks=_parse_lines(marks, "bookmark"),
                cut_lines=_parse_lines(split_lines, "split") or (), split_at_sections=use_sections,
                options=BookExportOptions(
                    export_type=container, aac_bitrate=bit_rate, normalization_type=NormalizationType.from_id(loudness_id),
                    reason_pauses=ReasonPauseTypes.get_by_id(pause_id), high_shelf=HighShelfEq.get_by_id(treble_id),
                    room_tone=RoomTone.get_by_id(tone_id), room_reverb_stereo=RoomReverbStereo.get_by_id(stereo_id),
                    use_break_sound_effect=effect, subdivide_phrases=subdivide,
                    min_segment_sentence_silence=(
                        MinSegmentSentenceSilence.ENABLED if sentence else MinSegmentSentenceSilence.DISABLED
                    ),
                    limit_silence_gaps=trim, limit_silence_gaps_duration=gap,
                    use_upsampler=use_upsampler,
                ),
            )
            choices = list(list_book_exports(session, run_id))
            first_result, last_result = results[0], results[-1]
            line_range = f"lines {first_result.line_start:,}–{last_result.line_end:,}"
            message = f"Exported {line_range}." if len(results) == 1 else f"Exported {len(results):,} files for {line_range}."
            missing = sum(result.missing_lines for result in results)
            if missing:
                message += f" Missing audio for {missing:,} lines."
            cached = download.move_resource_to_block_cache(first_result.path)
            return (
                gr.update(value=cached, visible=True),
                gr.update(value=reader_link(cached), visible=True),
                gr.update(choices=choices, value=first_result.export_id), show_status(message),
            )
        except (OSError, WebBookOutputError) as error:
            return gr.skip(), gr.skip(), gr.skip(), show_status(str(error))

    refresh.click(find_renderings, [book, runs], [runs, status], **private_event)
    book.change(find_renderings, [book, runs], [runs, status], **private_event)
    runs.change(
        select_rendering, [book, runs], [line_text, takes, take_audio, export_history, download, reader], **private_event,
    )
    review.click(read_line, [book, runs, line], [line_text, takes, status], **private_event)
    takes.change(play_take, [book, runs, line, takes], [take_audio, status], **private_event)
    export_history.input(open_export, [book, runs, export_history], [download, reader, status], **private_event)
    export.click(
        make_export, [
            book, runs, first, last, file_type, bookmarks, cuts, split_sections, bitrate, loudness, pauses, treble, room_tone, stereo,
            section_effect, subdivisions, sentence_pause, upsample, limit_gaps, max_gap,
        ], [download, reader, export_history, status], **private_event,
    )


def _parse_lines(text: str | None, label: str) -> tuple[int, ...] | None:
    if not text or not text.strip():
        return None
    try:
        return tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise WebBookOutputError(f"Enter {label} line numbers separated by commas.") from error
