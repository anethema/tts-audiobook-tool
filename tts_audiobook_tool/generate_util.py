from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import os
import time
from time import monotonic

import numpy as np

from tts_audiobook_tool import app_support, ask, text_util
from tts_audiobook_tool.app_types import Sound, SttConfig, SttVariant
from tts_audiobook_tool.app_support import app_hint_util, app_memory
from tts_audiobook_tool.checkpoint_writer import (
    CheckpointWriteError,
    CheckpointWriteQueue,
    SavedCheckpoint,
    StagedCheckpoints,
    choose_checkpoint_path,
    write_checkpoint,
)
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.generation_eta import GenerationEta
from tts_audiobook_tool.generation_events import (
    GenerationEvents,
    GenerationPhase,
    GenerationProgress,
    GenerationRunEnded,
    GenerationStarted,
    GenerationStats,
    ModelUnhealthy,
)
from tts_audiobook_tool.gen_timeout_util import make_backend_gen_timeout_tracker
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool import readiness
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.app_types.segment_transcript_data import SegmentTranscriptData
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.text_ops import language_util
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.validator import Validator
from tts_audiobook_tool.app_types.validation_result import MusicFailResult, SkippedResult, ExcessiveDurationResult, TranscriptResult, TrimmedResult, ValidationResult, WordErrorResult
from tts_audiobook_tool.transcriber import Transcriber
from tts_audiobook_tool.tts_models.voice_variation_warning import get_run_warnings


@dataclass(frozen=True)
class TtsModelError:
    message: str

@dataclass(frozen=True)
class SubBatch:
    """
    One model-inference call's worth of items, all of which use a single
    voice sample.

    `voice_selection_index` is None when `GenerateUtil.generate()` resolves
    the voice itself (a single line's assignment, or a model without voice
    samples).
    """
    voice_selection_index: int | None
    items: tuple[tuple[int, int], ...]

# Number of batch iterations applied per group in the windowed word-count
# ordering used to preserve a semblance of index monotonicity.
BATCH_ITERATIONS_PER_GROUP = 5

CHECKPOINT_DURATION_HISTORY_SIZE = 50


def make_consecutive_model_errors_reset_message(max_consecutive_model_errors: int) -> str:
    """Single source of the reason text for a consecutive-model-errors abort."""
    return (
        f"Too many consecutive TTS model errors ({max_consecutive_model_errors})"
    )


def make_oom_reset_message() -> str:
    """Single source of the reason text for an OOM abort."""
    return "TTS model ran out of memory"


class GenerateUtil:

    @staticmethod
    def generate_files(
            state: State,
            indices_set: set[int],
            batch_size: int,
            is_regen: bool,
            staging_dir: str = "",
            *,
            suppress_validation_passes: bool = False,
            guard_first_batch: bool = False,
    ) -> bool:
        """
        Subroutine for doing a series of audio generations to files
        Prints feedback at end of each item, and summary at end of loop.

        :param batch_size:
            When set to 1, batch mode is effectively disabled.
        :param staging_dir:
            When set, output is saved to this directory for later review instead
            of the project's segments directory: existing project segments are
            left untouched, redundant takes are pruned within the staging
            directory, and the project's generation range is not updated.
        :param suppress_validation_passes:
            Omit routine first-attempt validation passes from console output.
        :param guard_first_batch:
            Apply the watchdog from the first generate-and-validate batch.

        Returns:
            True if ended because interrupted (user pressed control-c)
            or aborted (OOM detected or too many consecutive model errors;
            an abort also emits a ModelUnhealthy event so the main process
            hard-resets the model worker)
        Raises CheckpointWriteError if accepted audio cannot be saved.
        """
        writer = (
            CheckpointWriteQueue(batch_size)
            if getattr(state.project, "checkpoint_format", "flac") == "aac"
            else None
        )
        with writer or nullcontext():
            try:
                return GenerateUtil._generate_files(
                    state, indices_set, batch_size, is_regen, staging_dir, suppress_validation_passes, writer,
                    guard_first_batch=guard_first_batch,
                )
            finally:
                Interrupts().clear()

    @staticmethod
    def _generate_files(
        state: State,
        indices_set: set[int],
        batch_size: int,
        is_regen: bool,
        staging_dir: str,
        suppress_validation_passes: bool,
        writer: CheckpointWriteQueue | None,
        *,
        guard_first_batch: bool,
    ) -> bool:
        project = state.project
        max_retries = project.max_retries
        stt_variant = state.prefs.stt_variant
        stt_config = state.prefs.stt_config
        showed_vram_warning = False
        generation_started = monotonic()

        GenerationEvents.emit(GenerationPhase("Preparing models"))

        warm_up_result = ModelManager.warm_up_models(state)
        if warm_up_result.should_stop:
            app_support.print_warm_up_result_stop(warm_up_result)
            if warm_up_result.error:
                app_memory.gc_ram_vram()
            return True

        # Do model readiness check now that model instance exists
        err = readiness.get_generate_blocker_text(state, verbose=True)
        if err:
            ask.ask_error(err)
            return True

        # Print warnings if any, including the voice-variation note for this run's voices
        warnings = get_run_warnings(state.project, Tts.get_instance(), indices_set)
        if warnings:
            # The TUI parses each console line independently, so style every line.
            message = "\n\n".join(
                "\n".join(Ansi.ITALICS + line for line in warning.split("\n"))
                for warning in warnings
            )
            print_feedback(message + "\n", no_preformat=True, extra_line=False)

        showed_vram_warning = app_memory.show_vram_memory_warning_if_necessary()

        # Make 'items' (tuple = phase group index, retry_count)
        sorted_indices = sorted(list(indices_set))
        GenerationEvents.emit(GenerationStarted(total=len(sorted_indices)))
        items: list[tuple[int, int]] = []
        for i in range(len(sorted_indices)):
            index = sorted_indices[i]
            items.append((index, 0))

        # Build a queue of rounds. A "round" is a group of sub-batches (a
        # "batch of batches"); each sub-batch is sent to the model as a single
        # call that can use only one voice sample.
        #
        # When batching with voice samples, lines are grouped into per-voice
        # sub-batches by their assigned voice (see make_multi_voice_rounds).
        # Otherwise the queue degenerates to the single-voice shape: a flat,
        # optionally word-count-bucketed list of items chunked into
        # sub-batches of batch_size, and generate() resolves each voice.
        is_cjk = language_util.is_cjk(state.project.language_code)
        voice_of_index: dict[int, int] | None = None
        if batch_size > 1:
            voice_values = ProjectVoiceUtil.get_voice_values(project, project.get_tts_model_type())
            if voice_values:
                voice_of_index = effective_voice_indices(
                    project.phrase_groups, sorted_indices, len(voice_values)
                )
        if voice_of_index is not None:
            queue = make_multi_voice_rounds(
                items, project.phrase_groups, voice_of_index, batch_size,
                sort_by_words=not is_cjk,
            )
        else:
            if batch_size > 1 and not is_cjk:
                items = bucket_items(items, project.phrase_groups, batch_size)
            queue = make_single_voice_rounds(items, batch_size)

        text_lengths = {
            index: max(1, sum(len(phrase.text) for phrase in project.phrase_groups[index].phrases))
            for index in sorted_indices
        }
        generation_eta = GenerationEta(text_lengths, batch_size)
        for queued_round in queue:
            for sub in queued_round:
                generation_eta.add_batch([index for index, _ in sub.items])

        # Metrics-related variables
        errored_indices: set[int] = set()
        num_failed = 0
        num_failed_music = 0
        num_improved = 0
        num_passed = 0
        num_retries = 0
        saved_indices: set[int] = set()
        saved_durations: dict[int, float] = {}
        saved_audio_seconds = 0.0
        speed_multiplier: float | None = None
        word_counts: dict[int, int] = {}
        preexisting_word_error_counts = project.sound_segments.get_word_error_counts_in_generate_range()
        best_word_error_counts = dict(preexisting_word_error_counts)
        gen_val_sum_time = 0
        start_time = time.time()
        consecutive_model_errors = 0
        max_consecutive_model_errors = 5
        # Recalculated once per completed batch; never time-interpolated.
        eta_seconds: float | None = None
        write_durations: deque[float] = deque(maxlen=CHECKPOINT_DURATION_HISTORY_SIZE)
        write_error = ""

        if staging_dir:
            publish = StagedCheckpoints(staging_dir).record
        else:
            def publish(path: Path, replaced: Path | None) -> None:
                project.sound_segments.record_published_segment(path, replaces=replaced)

        def save_take(
            index: int,
            phrase_group: PhraseGroup,
            result: ValidationResult,
            stt_info: SegmentTranscriptData | None,
            final_attempt: bool,
            did_improve: bool,
            quiet: bool,
        ) -> SavedCheckpoint:
            started = monotonic()
            error, path = GenerateUtil.save_sound_and_timing_json(
                state, phrase_group, index, result,
                is_real_time=False, voice_tag=getattr(result, "voice_tag", ""),
                stt_info=stt_info, dir_path=staging_dir or None, on_published=publish,
            )
            duration = result.sound.duration if not error and Path(path).is_file() else None
            return SavedCheckpoint(
                index, path, error, monotonic() - started, duration,
                not result.is_fail, final_attempt, did_improve, quiet,
            )

        def receive_save(saved: SavedCheckpoint) -> str:
            nonlocal write_error, num_passed, num_failed, num_improved, saved_audio_seconds
            if saved.error:
                errored_indices.add(saved.index)
                message = f"Couldn't save file: {saved.error} {saved.path}".rstrip()
                write_error = write_error or message
                return f"{COL_ERROR}{message}"

            write_durations.append(saved.elapsed)
            saved_indices.add(saved.index)
            if saved.passed or saved.final_attempt:
                errored_indices.discard(saved.index)
            num_passed += saved.passed
            num_failed += not saved.passed and saved.final_attempt
            num_improved += saved.improved
            # A retry contributes the duration of the retained take for its line.
            if saved.duration is not None:
                saved_audio_seconds += saved.duration - saved_durations.get(saved.index, 0.0)
                saved_durations[saved.index] = saved.duration
            if saved.quiet or saved.duration is None:
                return ""
            link = text_util.make_terminal_hyperlink(url=saved.path, text=Path(saved.path).name, is_file=True)
            return f"Saved: {COL_DIM}{link}{Ansi.RESET}"

        def receive_queued_saves(*, drain: bool = False, make_space: bool = False) -> None:
            if writer is not None:
                for saved in writer.collect(drain=drain, make_space=make_space):
                    if message := receive_save(saved):
                        printt(message)

        def emit_progress(current_indices: tuple[int, ...] = ()) -> None:
            num_errored = len(errored_indices)
            processed = num_passed + num_failed + num_errored
            GenerationEvents.emit(
                GenerationProgress(
                    processed=processed,
                    remaining=len(sorted_indices) - processed,
                    total=len(sorted_indices),
                    current_indices=current_indices,
                    passed=num_passed,
                    failed=num_failed,
                    errored=num_errored,
                    retries=num_retries,
                    eta_seconds=eta_seconds,
                    speed_multiplier=speed_multiplier,
                    saved=num_passed + num_failed,
                )
            )

        Interrupts().set("generating")
        did_interrupt = False
        did_abort_model_errors = False
        model_reset_reason = ""

        # Audiobook generation is a top-level generation run. Start from a
        # fresh rolling-continuation context, independent of any prior run.
        Tts.clear_continuation()

        # Loop through the queue of rounds. Retries (re-added items) are
        # collected and processed at the head of the queue, as their own
        # round(s), before any further items from the main queue.
        pending_retries: list[tuple[int, int]] = []
        current_round: list[SubBatch] = []
        last_voice: int | None = None

        gen_timeout_tracker = make_backend_gen_timeout_tracker(
            guard_first_batch=guard_first_batch,
        )

        while True:
            receive_queued_saves()
            if write_error:
                break
            if Interrupts().did_interrupt:
                did_interrupt = True
                break

            # Start a new round if needed (retries take priority over the queue)
            if not current_round:
                if pending_retries:
                    current_round = make_retry_round(pending_retries, voice_of_index, batch_size)
                    pending_retries = []
                    generation_eta.start_retry_round()
                elif queue:
                    current_round = queue.pop(0)
                else:
                    break

            sub = current_round.pop(0)

            # Make parallel arrays from 'sub'
            indices = [item[0] for item in sub.items]
            batch_work = generation_eta.start_batch(indices)
            retry_counts = [item[1] for item in sub.items]
            num_retries += sum(retry_count > 0 for retry_count in retry_counts)

            # A rolling-continuation context is per voice sample, so do not
            # carry it across a voice sample change mid-run.
            if (
                sub.voice_selection_index is not None
                and last_voice is not None
                and sub.voice_selection_index != last_voice
            ):
                Tts.clear_continuation()
            last_voice = sub.voice_selection_index

            if not showed_vram_warning:
                b = app_memory.show_vram_memory_warning_if_necessary()
                if b:
                    print("\a", end="")
                    showed_vram_warning = True

            # Print item info and publish a structured progress snapshot.
            # The snapshot carries the previous batch's ETA so the header
            # does not blank the estimate during this batch's inference.
            GenerationEvents.emit(GenerationPhase("Generating audio"))
            emit_progress(tuple(indices))
            heading_voice_index = sub.voice_selection_index
            if heading_voice_index is None and len(indices) == 1:
                heading_voice_index = GenerateUtil.get_line_voice_index(
                    project, project.phrase_groups, indices[0]
                )
            GenerateUtil.print_batch_heading(
                indices=indices,
                voice_index=heading_voice_index,
                show_divider=True,
            )

            # Generate and validate
            gen_start_time = monotonic()
            with gen_timeout_tracker.scope() as gen_timeout_guard:
                try:
                    results = GenerateUtil.generate_and_validate_batch(
                        state=state,
                        indices=indices,
                        phrase_groups=project.phrase_groups,
                        stt_variant=stt_variant, stt_config=stt_config,
                        force_random_seed=is_regen or any(count > 0 for count in retry_counts),
                        is_realtime=False,
                        voice_selection_index=sub.voice_selection_index,
                    )
                except RuntimeError as error:
                    # Some GPU backends raise during lazy initialization, before
                    # synthesis/Whisper can return a per-line OOM result.
                    message = f"{type(error).__name__}: {error}"
                    if type(error).__name__ != "OutOfMemoryError" and "out of memory" not in str(error).lower():
                        raise
                    printt(f"{COL_ERROR}{message}")
                    results = [TtsModelError(message)]
            if gen_timeout_guard.did_time_out:
                # The watchdog already reported the timeout and requested the
                # worker reset; abort the run regardless of any pending cancel.
                printt(f"{COL_ERROR}Aborting generation run")
                Tts.clear_continuation()
                did_interrupt = True
                break
            batch_gen_val_time = monotonic() - gen_start_time
            gen_val_sum_time += batch_gen_val_time

            # Check for OOM in results and break early if detected
            if any(GenerateUtil.is_error_result_oom(r) for r in results):
                # The per-segment error line already printed the raw error, and the
                # reset reason below is shown by the hard-reset UI; print nothing more.
                model_reset_reason = make_oom_reset_message()
                Tts.clear_continuation()
                did_interrupt = True
                break

            # Process and print results # TODO: separate biz n print logic
            re_adds: list[tuple[int, int]] = []

            for i, result in enumerate(results):

                is_model_error = isinstance(result, TtsModelError)
                if isinstance(result, TtsModelError):
                    error_string = result.message
                    validation_result = None
                elif isinstance(result, str):
                    error_string = result
                    validation_result = None
                else:
                    error_string = ""
                    validation_result = result

                index = indices[i]
                phrase_group = project.phrase_groups[index]
                stt_info: SegmentTranscriptData | None = None
                should_show_viz = False
                quiet_pass = (
                    suppress_validation_passes
                    and isinstance(validation_result, TranscriptResult)
                    and not validation_result.is_fail
                    and retry_counts[i] == 0
                )
                message_lines = []

                retry_string = f" (retry #{retry_counts[i]})" if retry_counts[i] else ""
                item_line = f"{COL_ACCENT}Line {index + 1}{retry_string}:" # {COL_DEFAULT}{text_string}"
                message_lines.append(item_line)

                new_retry_count = retry_counts[i] + 1


                if error_string:
                    # Model error
                    if is_model_error:
                        consecutive_model_errors += 1
                    else:
                        consecutive_model_errors = 0
                    did_abort_model_errors = consecutive_model_errors >= max_consecutive_model_errors
                    val_line = f"{COL_ERROR}Error: {error_string}"
                    if did_abort_model_errors:
                        val_line += f"; {COL_ERROR}too many consecutive TTS model errors"
                    elif new_retry_count > max_retries:
                        errored_indices.add(index)
                        val_line += f"; {COL_ERROR}max retries reached"
                    else:
                        re_adds.append((index, new_retry_count))
                        val_line += f"; {COL_DEFAULT}will retry"
                    message_lines.append(val_line)

                elif validation_result:
                    consecutive_model_errors = 0

                    val_line = "" if quiet_pass else validation_result.get_ui_message_with_extras()
                    validation_word_error_count: int | None = None
                    if isinstance(validation_result, TranscriptResult):
                        validation_word_error_count = SegmentTranscriptUtil.make_generation_word_error_count(validation_result)
                    previous_word_error_count = best_word_error_counts.get(index, None)
                    did_improve = (
                        (is_regen or retry_counts[i] > 0)
                        and validation_word_error_count is not None
                        and previous_word_error_count is not None
                        and validation_word_error_count < previous_word_error_count
                    )
                    if validation_word_error_count is not None and (
                        previous_word_error_count is None or validation_word_error_count < previous_word_error_count
                    ):
                        best_word_error_counts[index] = validation_word_error_count

                    # Modify validation ui message if needed
                    if not validation_result.is_fail:
                        if is_regen:
                            val_line = val_line.replace("Passed", f"{COL_OK}Passed")
                        else:
                            if new_retry_count > 1:
                                val_line = val_line.replace("Passed", f"{COL_OK}Passed on retry")
                    else:
                        if is_regen:
                            if isinstance(validation_result, WordErrorResult):
                                preexisting_count = preexisting_word_error_counts.get(index, None)
                                new_count = validation_result.num_errors
                                if preexisting_count is not None and validation_result.num_errors < preexisting_count:
                                    val_line += f" (improved from {preexisting_count} to {new_count} word errors)"
                                    preexisting_word_error_counts[index] = new_count
                        elif did_improve and isinstance(validation_result, WordErrorResult):
                            val_line += f" (improved from {previous_word_error_count} to {validation_result.num_errors} word errors)"

                    # Failed or not
                    if validation_result.is_fail:
                        Tts.clear_continuation()
                        if new_retry_count > max_retries:
                            val_line += f"; {COL_ERROR}max retries reached, tagging as failed"
                        else:
                            re_adds.append((index, new_retry_count))
                            val_line += f"; {COL_DEFAULT}will retry"
                        if isinstance(validation_result, MusicFailResult):
                            num_failed_music += 1

                    # Print STT info
                    if isinstance(validation_result, TranscriptResult):
                        stt_info = SegmentTranscriptUtil.from_validation_result(
                            project=project,
                            phrase_group=phrase_group,
                            index=index,
                            validation_result=validation_result
                        )
                        should_show_viz = (
                            isinstance(validation_result, (MusicFailResult, ExcessiveDurationResult))
                            or (isinstance(validation_result, WordErrorResult) and validation_result.num_errors > 0)
                        )

                    if not quiet_pass:
                        message_lines.append(val_line + Ansi.RESET)
                    else:
                        # A routine pass can still report a trim or truncation warning.
                        warnings = (
                            validation_result.get_ui_message() if isinstance(validation_result, TrimmedResult) else "",
                            validation_result.get_possible_truncation_ui_message(),
                            validation_result.get_generated_trim_ui_message(),
                            validation_result.get_trailing_token_noise_trim_ui_message(),
                        )
                        message_lines.extend(warning + Ansi.RESET for warning in warnings if warning)

                    # Update word count metrics
                    word_counts[index] = phrase_group.num_words

                    # Save
                    save = partial(
                        save_take, index, phrase_group, validation_result, stt_info,
                        new_retry_count > max_retries, did_improve, quiet_pass,
                    )
                    if writer is None:
                        if message := receive_save(save()):
                            message_lines.append(message)
                    else:
                        receive_queued_saves(make_space=True)
                        if not write_error:
                            writer.submit(save)

                if quiet_pass and len(message_lines) == 1:
                    if write_error:
                        break
                    continue
                printt("\n".join(message_lines))

                if validation_result and not quiet_pass:
                    printt()
                    if stt_info is not None:
                        printt(
                            text_util.combine_ansi_lines(
                                SegmentTranscriptUtil.make_stt_details_lines(
                                    stt_info,
                                    should_show_diff=should_show_viz,
                                )
                            )
                        )
                    else:
                        # Just print source text, in a similar style
                        printt(f"{COL_DEFAULT}Source text: {COL_DIM_ITALICS}{phrase_group.presentable_text.strip()}")

                if i < len(results) - 1:
                    printt()

                if did_abort_model_errors or write_error:
                    break

            receive_queued_saves()

            # Include synchronous saves, pruning and any time waiting for
            # room in the writer queue. Error batches may exit early.
            if (
                not write_error
                and len(results) == len(indices)
                and not any(isinstance(result, (TtsModelError, str)) for result in results)
            ):
                generation_eta.record_duration(batch_work, monotonic() - gen_start_time)

            if re_adds:
                # Pending retries are processed at the head of the queue,
                # as the next round(s) before any further items from the
                # main queue.
                pending_retries.extend(re_adds)
                for index, _ in re_adds:
                    voice = voice_of_index[index] if voice_of_index is not None else 0
                    generation_eta.add_retry(index, voice)

            eta_seconds = generation_eta.seconds_remaining()
            if writer is not None:
                if write_durations:
                    seconds_per_write = sum(write_durations) / len(write_durations)
                    drain_eta = writer.pending * seconds_per_write
                    if not generation_eta.remaining_items:
                        eta_seconds = drain_eta
                    elif eta_seconds is not None:
                        # The writer can overlap synthesis, except for the
                        # writes submitted by the final sub-batch.
                        if queue:
                            last_batch_size = len(queue[-1][-1].items)
                        elif pending_retries:
                            last_batch_size = generation_eta.last_pending_retry_batch_size
                        else:
                            last_batch_size = len(current_round[-1].items)
                        all_writes_eta = drain_eta + generation_eta.remaining_items * seconds_per_write
                        eta_seconds = max(eta_seconds + last_batch_size * seconds_per_write, all_writes_eta)
                elif not generation_eta.remaining_items and writer.pending:
                    eta_seconds = None
            elapsed = monotonic() - generation_started
            speed_multiplier = saved_audio_seconds / elapsed if saved_audio_seconds > 0 and elapsed > 0 else None
            emit_progress()

            if did_abort_model_errors:
                printt()
                GenerateUtil.print_consecutive_model_errors_message(max_consecutive_model_errors)
                model_reset_reason = make_consecutive_model_errors_reset_message(max_consecutive_model_errors)
                Tts.clear_continuation()
                did_interrupt = True
                break

            # Memory usage is displayed by the generation header, not the
            # console; keep only the blank-line separator between batches.
            printt()

        if writer is not None:
            if writer.pending:
                GenerationEvents.emit(GenerationPhase("Saving audio"))
                eta_seconds = writer.pending * sum(write_durations) / len(write_durations) if write_durations else None
                emit_progress()
                receive_queued_saves(drain=True)
            did_interrupt = did_interrupt or Interrupts().did_interrupt
            eta_seconds = None if did_interrupt or write_error else 0.0
            elapsed = monotonic() - generation_started
            speed_multiplier = saved_audio_seconds / elapsed if saved_audio_seconds > 0 and elapsed > 0 else None
            emit_progress()

        # Update the persisted queue once per generation run, based only on files
        # that were actually saved successfully. Staged output is not part of
        # the project until the user accepts it.
        if saved_indices and not staging_dir:
            selected_indices = ProjectUtil.get_indices_to_generate(project)
            remaining_indices = selected_indices - saved_indices
            updated_range_string = RangeStringUtil.make_ranges_string(
                remaining_indices,
                len(project.phrase_groups),
            )
            if updated_range_string != project.generate_range_string:
                project.generate_range_string = updated_range_string
                save_error = project.save()
                if save_error:
                    ask.ask_error(
                        f"Couldn't save updated generation range: {save_error}",
                    )

        did_interrupt = did_interrupt or Interrupts().did_interrupt
        if model_reset_reason:
            GenerationEvents.emit(ModelUnhealthy(reason=model_reset_reason))

        # Print summary info
        elapsed_line = f"Elapsed: {duration_string(time.time() - start_time)}"
        divider = "-" * len(elapsed_line)
        message = f"{COL_ACCENT}{divider}\n{COL_DEFAULT}{elapsed_line}\n"

        ok = str(num_passed)
        if num_passed == len(sorted_indices):
            ok += " (all)"
        message += f"Lines saved: {COL_OK}{ok}{COL_DEFAULT}\n"
        if Stt.has_instance() and ModelManager.has_yamnet_detector():
            message += f"Num retries triggered due to detected music: {num_failed_music}\n"

        message += f"Num retries: {num_retries}\n"
        if num_improved:
            message += f"Lines improved on retry: {COL_OK}{num_improved}{COL_DEFAULT}\n"

        col = COL_ACCENT if num_failed else ""
        if num_failed:
            message += f"Lines saved, but with excess word errors: {col}{num_failed}{COL_DEFAULT}\n"
        if errored_indices:
            message += f"Lines failed to generate: {COL_ERROR}{len(errored_indices)}{COL_DEFAULT}\n"
        if DEV:
            message += f"Num words: {sum(word_counts.values())}\n"
            message += f"Gen/val elapsed: {duration_string(gen_val_sum_time)}\n"
        quick_generation_completed_cleanly = (
            is_regen
            and not did_interrupt
            and not write_error
            and num_failed == 0
            and not errored_indices
        )
        # The batch loop is over: publish the run-end boundary before the
        # trailing summary block (whose first line is a dash rule that a
        # Textual consumer promotes to a full-width closing divider).
        GenerationEvents.emit(GenerationRunEnded())
        if not quick_generation_completed_cleanly:
            printt(message)

        # Prevent rolling-continuation state from leaking into a later run when
        # this run ends successfully without a paragraph/section break.
        Tts.clear_continuation()

        if write_error:
            raise CheckpointWriteError(write_error)
        return did_interrupt

    @staticmethod
    def generate_and_validate_batch(
        state: State,
        indices: list[int],
        phrase_groups: list[PhraseGroup],
        stt_variant: SttVariant,
        stt_config: SttConfig,
        force_random_seed: bool,
        is_realtime: bool,
        is_skip_reason_buffer: bool=False,
        voice_selection_index: int | None=None
    ) -> list[ValidationResult | str | TtsModelError]:
        """
        Generates and validates a batch of prompts from the Project text.
        Prints updates.
        Returns a list of results (ValidationResult or error string).

        :param indices:
            When length is 1, batch mode is disabled
        :param voice_selection_index:
            When set, all items in the batch are generated with this voice
            sample (a batch must be voice-homogeneous). When None, the voice
            is resolved by the existing logic inside `generate()`.
        """

        project = state.project
        save_debug_files = state.prefs.save_debug_files

        # Set print color to dim during any model inference printouts
        print(f"{COL_DIM}", end="")

        # Generate:
        gen_start_time = time.time()
        gen_results = GenerateUtil.generate(
            project=project,
            indices=indices,
            phrase_groups=phrase_groups,
            force_random_seed=force_random_seed,
            is_realtime=is_realtime,
            save_debug_files=save_debug_files,
            print_generation_request=True,
            print_params=True,
            voice_selection_index=voice_selection_index
        )
        printt() # Restore print color, print blank line

        # Print speed info
        print_speed_info(time.time() - gen_start_time, gen_results)

        # Should skip or not
        skip_reason = Stt.should_skip(state, is_skip_reason_buffer)
        if not skip_reason:
            GenerationEvents.emit(GenerationPhase("Transcribing and validating"))
            printt(f"{COL_DEFAULT}Transcribing audio...", end="") # gets overwritten
        else:
            GenerationEvents.emit(GenerationPhase("Processing generated audio"))

        val_start_time = time.time()
        results: list[ValidationResult | str | TtsModelError] = []

        for i, gen_result in enumerate(gen_results):

            index = indices[i]

            if isinstance(gen_result, TtsModelError):
                err = gen_result.message
                printt(f"{COL_ERROR}Error: {err}")
                results.append(gen_result)
                continue

            if isinstance(gen_result, str):
                err = gen_result
                printt(f"{COL_ERROR}Error: {err}")
                results.append(err)
                continue

            sound, start_trim_time, end_trim_time, original_duration, token_noise_trim_time, voice_tag = gen_result

            if skip_reason:
                validation_result = SkippedResult(sound=sound, message=skip_reason)
                validation_result.generated_start_trim_time = start_trim_time
                validation_result.generated_end_trim_time = end_trim_time
                validation_result.generated_trim_original_duration = original_duration
                validation_result.trailing_token_noise_trim_time = token_noise_trim_time
                validation_result.voice_tag = voice_tag
                results.append((validation_result))
                continue

            # Transcribe
            gen_result = Transcriber.transcribe_to_words(
                sound, project.language_code, stt_variant, stt_config
            )
            if isinstance(gen_result, str):
                err = gen_result
                results.append(err)
                continue
            transcribed_words = gen_result

            # Validate
            text = phrase_groups[ indices[i] ].spoken_text(project.language_code)
            validation_result = Validator.validate(
                sound, text, transcribed_words, project.language_code, strictness=project.strictness
            )
            validation_result.generated_start_trim_time = start_trim_time
            validation_result.generated_end_trim_time = end_trim_time
            validation_result.generated_trim_original_duration = original_duration
            validation_result.trailing_token_noise_trim_time = token_noise_trim_time
            validation_result.voice_tag = voice_tag
            results.append(validation_result)

            if save_debug_files and isinstance(validation_result, TrimmedResult):
                GenerateUtil.save_debug_sound(
                    project=project,
                    index=index,
                    label="trim",
                    sound=validation_result.sound,
                    is_realtime=is_realtime
                )

        if not skip_reason:
            message = f"{COL_DEFAULT}Transcribed audio in {(time.time() - val_start_time):.1f}s"
            printt(f"{Ansi.LINE_HOME}{message}")

        printt()

        return results

    @staticmethod
    def generate(
            project: Project,
            indices: list[int],
            phrase_groups: list[PhraseGroup],
            force_random_seed: bool,
            is_realtime: bool,
            save_debug_files: bool,
            print_generation_request: bool = False,
            print_params: bool = False,
            voice_selection_index: int | None = None
        ) -> list[tuple[Sound, float | None, float | None, float, float | None, str] | str | TtsModelError]:
        """
        Core audio generation function.

        Returns a list of "results" (same length as that of prompt_text),
        where each result is either a generated Sound or an error string.
        The Sound is trimmed for silence and peak-normalized.

        param indices:
            Use a one-element list to not generate in batch mode
        param voice_selection_index:
            When set, all items are generated with this voice sample (the
            batch must be voice-homogeneous). When None, a single item uses
            its line's assigned voice.
        """

        if len(indices) == 0:
            raise ValueError("Indices cannot be empty")
        if len(indices) > 1 and not project.get_tts_model_type().can_batch():
            raise ValueError("Logic error - Model does not support batching")

        # Make prompts (parallel list to indices)
        prompts = []
        for index in indices:
            prompt = GenerateUtil.phrase_group_to_prompt(phrase_groups[index], project)
            prompts.append(prompt)

        # Generate
        voice_tag = ""
        if Tts.get_active_type().id == "none":
            result = "No active TTS model"
        else:
            voice_values = ProjectVoiceUtil.get_voice_values(project, project.get_tts_model_type())
            if voice_selection_index is None:
                if not voice_values:
                    voice_selection_index = 0
                elif len(indices) > 1:
                    raise ValueError(
                        "Logic error - A multi-item batch must be split into "
                        "per-voice sub-batches with explicit "
                        "voice_selection_index values (see make_multi_voice_rounds)"
                    )
                else:
                    # Invalid (presumably stale) assignments fall back to voice 0.
                    voice_selection_index = ProjectVoiceUtil.get_effective_voice_index(
                        phrase_groups[indices[0]].voice_index, len(voice_values),
                    )
            voice_tag = Tts.get_voice_tag_for_selection_index(project, voice_selection_index)
            delivery_cues = [phrase_groups[index].delivery for index in indices]
            delivery_kwargs = {"delivery_cues": delivery_cues} if any(delivery_cues) else {}
            result = Tts.generate_using_project(
                project,
                prompts,
                force_random_seed,
                voice_selection_index=voice_selection_index,
                print_generation_request=print_generation_request,
                print_params=print_params,
                **delivery_kwargs,
            )

        # `result` is either n generated Sounds or a single error string
        if isinstance(result, str):
            Tts.clear_continuation()
            return [TtsModelError(result) for _ in range(len(prompts))] # return n model errors

        sounds = [result] if isinstance(result, Sound) else result

        results: list[tuple[Sound, float | None, float | None, float, float | None, str] | str | TtsModelError] = [] # TODO: Revisit

        for i, sound in enumerate(sounds):

            if isinstance(sound, str):
                result = TtsModelError(sound)
                results.append(result)
                Tts.clear_continuation()
                Tts.clear_continuation_if_reason(phrase_groups[indices[i]].last_reason)
                continue

            if sound.data.size == 0:
                result = "Model output is empty, discarding"
                Tts.clear_continuation()
            elif np.sum(np.isnan(sound.data)) > 0:
                result = "Model outputted NaN, discarding"
                Tts.clear_continuation()
            else:
                # Save "raw" output
                if save_debug_files:
                    GenerateUtil.save_debug_sound(project, indices[i], "raw", sound, is_realtime=is_realtime)

                # Trim silence ends and peak-normalize
                sound, start_trim_time, end_trim_time, original_duration = SoundPipeline.apply_generate_post_processing_with_info(sound)
                token_noise_trim_time = None

                # Trim model-specific short token-like trailing artifacts.
                # Done before STT transcription so timestamps match the final audio.
                if sound.data.size > 0 and Tts.get_model_support(project).should_trim_trailing_token_noise(project, Tts.get_instance_if_exists()):
                    pre_token_noise_trim_duration = sound.duration
                    if save_debug_files:
                        GenerateUtil.save_debug_sound(project, indices[i], "pre_token_noise_trim", sound, is_realtime=is_realtime)
                    trimmed_sound = SoundExtraUtil.trim_trailing_token_noise(sound)
                    if len(trimmed_sound.data) != len(sound.data):
                        sound = trimmed_sound
                        token_noise_trim_time = pre_token_noise_trim_duration - sound.duration
                        if save_debug_files:
                            GenerateUtil.save_debug_sound(project, indices[i], "post_token_noise_trim", sound, is_realtime=is_realtime)

                # Internal silence gap limiting is not done here: it happens when
                # concatenating, so re-concatenating saved audio with a different
                # threshold doesn't need a regeneration pass. Saved audio is
                # therefore un-cut, which is also what STT timestamps are recorded
                # against.

                if sound.data.size == 0:
                    result = "Model output is silence, discarding"
                    Tts.clear_continuation()
                else:
                    result = (sound, start_trim_time, end_trim_time, original_duration, token_noise_trim_time, voice_tag)

            results.append(result)
            Tts.clear_continuation_if_reason(phrase_groups[indices[i]].last_reason)

        return results

    @staticmethod
    def get_error_result_message(result: ValidationResult | str | TtsModelError) -> str:
        if isinstance(result, TtsModelError):
            return result.message
        if isinstance(result, str):
            return result
        return ""

    @staticmethod
    def is_error_result_oom(result: ValidationResult | str | TtsModelError) -> bool:
        return is_oom_error_message(GenerateUtil.get_error_result_message(result))

    @staticmethod
    def print_consecutive_model_errors_message(max_consecutive_model_errors: int) -> None:
        printt(
            f"{COL_ERROR}Stopping generation: "
            f"Too many consecutive TTS model errors ({max_consecutive_model_errors})."
        )
        printt()

    @staticmethod
    def phrase_group_to_prompt(phrase_group: PhraseGroup, project: Project) -> str:

        prompt = phrase_group.spoken_text(project.language_code)
        return prompt

    @staticmethod
    def save_sound_and_timing_json(
        state: State,
        phrase_group: PhraseGroup,
        index: int,
        validation_result: ValidationResult,
        is_real_time: bool,
        voice_tag: str = "",
        stt_info: SegmentTranscriptData | None = None,
        dir_path: str | None = None,
        sound: Sound | None = None,
        on_published: Callable[[Path, Path | None], None] | None = None,
    ) -> tuple[str, str]:
        """
        Saves sound segment and timing info json
        Returns error string (if any), saved file path

        :param dir_path:
            Output directory; defaults to the project's realtime or segments
            directory. The sidecar json is always saved alongside the sound.
        :param sound:
            Audio to save instead of `validation_result.sound`. Realtime uses
            this to save what was actually played (it writes no sidecar).
        :param on_published:
            Receives the completed checkpoint path and any same-name older
            take after the audio and timing files have been published.
        """

        project = state.project
        in_project = not is_real_time and not dir_path
        directory = Path(dir_path or (project.realtime_path if is_real_time else project.sound_segments_path))
        use_aac = not is_real_time and getattr(project, "checkpoint_format", "flac") == "aac"

        file_name = SoundSegmentUtil.make_file_name(
            index=index,
            phrase_group=phrase_group,
            project=project,
            validation_result=validation_result,
            tts_model_type=project.get_tts_model_type().value,
            is_real_time=is_real_time,
            voice_tag=voice_tag,
            suffix=".m4a" if use_aac else ".flac",
        )
        path = directory / file_name
        audio = validation_result.sound if sound is None else sound
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if is_real_time:
                return SoundFileUtil.save_flac(audio, str(path)), str(path)

            if isinstance(validation_result, TranscriptResult) and stt_info is None:
                stt_info = SegmentTranscriptUtil.from_validation_result(
                    project=project, phrase_group=phrase_group, index=index, validation_result=validation_result,
                )

            path, replaced = choose_checkpoint_path(path)
            bitrate = state.prefs.aac_bitrate if use_aac else AAC_BITRATE_DEFAULT
            error = write_checkpoint(path, audio, stt_info, bitrate)
            if error:
                return error, str(path)

            if on_published is not None:
                on_published(path, replaced)
            elif in_project:
                project.sound_segments.record_published_segment(path, prune=False, replaces=replaced)
        except OSError as error:
            return make_error_string(error), str(path)

        return "", str(path)

    @staticmethod
    def get_line_voice_index(
        project: Project, phrase_groups: list[PhraseGroup], index: int
    ) -> int | None:
        """
        The voice sample index a single line will be generated with (same
        fallback as `generate()`), or None for a model without voice samples.
        """
        voice_values = ProjectVoiceUtil.get_voice_values(project, project.get_tts_model_type())
        if not voice_values:
            return None
        return ProjectVoiceUtil.get_effective_voice_index(
            phrase_groups[index].voice_index, len(voice_values)
        )

    @staticmethod
    def print_batch_heading(
        indices: list[int],
        voice_index: int | None = None,
        show_divider: bool = True,
    ) -> None:

        line_noun = make_noun("line", "lines", len(indices))
        if len(indices) <= 4:
            indices_string = ", ".join(str(item + 1) for item in indices)
        else:
            index_strings = [str(item + 1) for item in indices[:3]]
            num_more = len(indices) - 3
            indices_string = f"{', '.join(index_strings)}, +{num_more} more"
        processing_string = f"Processing {line_noun} {indices_string}"
        if voice_index is not None:
            processing_string += f" {COL_DIM}(voice {voice_index + 1}){COL_DEFAULT}"

        if show_divider:
            print_accent_separator(processing_string)
        else:
            printt(f"{COL_ACCENT}{processing_string}")
            printt()

    @staticmethod
    def save_debug_sound(project: Project, index: int, label: str, sound: Sound, is_realtime: bool):
        """
        Save a sound as a labeled FLAC file for debugging purposes.
        Written to the project's realtime directory if `is_realtime`, otherwise to the
        sound segments directory.
        """
        dir_path = project.realtime_path if is_realtime else project.sound_segments_path
        os.makedirs(dir_path, exist_ok=True)
        index_string = str(index + 1).zfill(5)
        timestamp = SoundSegmentUtil.make_timestamp_string()

        # Use same start of filename as sound segment files
        if is_realtime:
            file_name = f"[{timestamp}] [{index_string}] [{label}].flac"
        else:
            file_name = f"[{index_string}] [{timestamp}] [{label}].flac"
        path = os.path.join(dir_path, file_name)
        _ = SoundFileUtil.save_flac(sound, path)

    @staticmethod
    def do_generate_using_project_and_state(state: State) -> None:
        """ Generate queued project items and run configured completion actions. """

        indices = ProjectUtil.get_selected_indices_not_generated(state.project)
        if not indices:
            return

        # Show pre-inference hint/warning if necessary
        should_continue = app_hint_util.show_pre_inference_hints(state.prefs, state.project)
        if not should_continue:
            return

        # Main-process UI flow only; see the lazy-import note above.
        from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
        voice_indices = ProjectVoiceUtil.get_job_voice_indices(state.project, indices)
        if not VoiceMenuShared.validate_voices(state, voice_indices=voice_indices):
            return

        from tts_audiobook_tool.textual.generation_app import run_generation_app
        generation_result = run_generation_app(
            state=state,
            indices=indices,
            batch_size=ProjectVoiceUtil.get_batch_size(state.project),
        )
        if not generation_result.completed:
            return

        app_support.play_done_sound()

        if state.project.gen_auto_concat:
            printt()
            ConcatUtil.auto_concat_after_generation(state)

# ---

def print_speed_info(gen_elapsed: float, gen_results: list) -> None:

    # Elapsed
    message = f"Generated audio in {gen_elapsed:.1f}s"

    # Cumulative duration, speed
    num_sounds = 0
    cum_duration = 0.0
    for item in gen_results:
        if isinstance(item, tuple) and isinstance(item[0], Sound):
            sound = item[0]
            num_sounds += 1
            cum_duration += sound.duration
        elif isinstance(item, Sound):
            num_sounds += 1
            cum_duration += item.duration
    if num_sounds == len(gen_results):
        message += f"; sound duration: {cum_duration:.1f}s; "
        gen_elapsed = max(gen_elapsed, 0.1)
        raw_speed = cum_duration / gen_elapsed
        speed = min(raw_speed, 99.9)
        speed_str = f"{speed:.1f}"
        if speed == 99.9:
            speed_str += "+"
        speed_str += "x"
        message += f"speed: {speed_str}"
        # Publish structured stats only when every result produced audio,
        # mirroring the console message; a partial batch has no meaningful
        # realtime factor (failed results contribute no audio duration).
        realtime_factor = gen_elapsed / cum_duration if cum_duration > 0 else 0.0
        GenerationEvents.emit(
            GenerationStats(
                generation_seconds=gen_elapsed,
                audio_seconds=cum_duration,
                realtime_factor=realtime_factor,
                speed_factor=raw_speed,
            )
        )

    print(message)

def print_eta(saved_elapsed: list[float], num_left) -> None:

    MIN_SAMPLES = 5
    if len(saved_elapsed) < MIN_SAMPLES or num_left <= 0:
        return

    MAX_SAMPLES = 50
    i_start = len(saved_elapsed) - MAX_SAMPLES
    i_start = max(i_start, 0)
    i_end = len(saved_elapsed)
    sum = 0
    for i in range(i_start, i_end):
        elapsed = saved_elapsed[i]
        sum += elapsed
    num_samples = i_end - i_start
    avg = sum / num_samples
    eta = num_left * avg
    printt(f"Est. time remaining: {duration_string(eta)}")

def bucket_items(
        items: list[tuple[int, int]],
        phrase_groups: list[PhraseGroup],
        batch_size: int) -> list[tuple[int, int]]:
    """
    Returns a copy of `items`, sorted by phrase group word count in an alternating
    descending/ascending pattern. Reason for sorting in groups is to preserve the
    ordering of items to some extent. Does not "bucket" per-se.

    Params:
        items: list of tuples (phase group index, retry_count),
               which should be sorted by index
    """

    # Make copy of items, but with added element for word count
    triplets: list[tuple[int, int, int]] = []

    for index, retry_count in items:
        phrase_group = phrase_groups[index]
        num_words = phrase_group.num_words
        triplet = (index, retry_count, num_words)
        triplets.append(triplet)

    group_size = batch_size * BATCH_ITERATIONS_PER_GROUP
    is_descending = True

    # Create a copy of triplets which is sorted in groups of size group_size,
    # alternating between descending and ascending.
    grouped_triplets = []
    for i in range(0, len(triplets), group_size):
        group = triplets[i:i + group_size]
        group.sort(key=lambda x: x[2], reverse=is_descending)
        grouped_triplets.extend(group)
        is_descending = not is_descending

    # "Extract" the index and retry count elements
    sorted_items = []
    for index, retry_count, _ in grouped_triplets:
        sorted_items.append( (index, retry_count) )
    return sorted_items

def effective_voice_indices(
        phrase_groups: list[PhraseGroup],
        indices: list[int],
        num_voice_values: int,
) -> dict[int, int]:
    """
    Maps each item index to the voice sample index that `GenerateUtil.generate`
    will actually use for that line.

    A stale out-of-range per-line selection falls back to voice sample index 0.
    """
    return {
        index: ProjectVoiceUtil.get_effective_voice_index(phrase_groups[index].voice_index, num_voice_values)
        for index in indices
    }

def make_single_voice_rounds(
        items: list[tuple[int, int]],
        batch_size: int,
) -> list[list[SubBatch]]:
    """
    Wraps a flat (possibly word-count-bucketed) queue into a list of rounds,
    each holding a single voice-less sub-batch of up to `batch_size` items.
    The voice is resolved inside `GenerateUtil.generate()` (the single line's
    assignment, or voice 1 for a model without voice samples).
    """
    return [
        [SubBatch(voice_selection_index=None, items=tuple(items[i:i + batch_size]))]
        for i in range(0, len(items), batch_size)
    ]

def make_multi_voice_rounds(
        items: list[tuple[int, int]],
        phrase_groups: list[PhraseGroup],
        voice_of_index: dict[int, int],
        batch_size: int,
        sort_by_words: bool,
) -> list[list[SubBatch]]:
    """
    Builds a queue of rounds for a batched project with voice samples.
    Each round is a group of per-voice sub-batches ("batches of batches");
    each sub-batch holds at most `batch_size` items that all share one voice
    sample, so each can be sent to the model as a single call.

    The single-speaker windowing principle is applied within practical limits:
    items are walked in index-sorted windows of
    `batch_size * BATCH_ITERATIONS_PER_GROUP * num_voices` items, so each
    voice keeps a comparable per-window share as the single-speaker path.
    Within a window, each voice's items are sorted by word count (alternating
    descending/ascending per window) to pack similar-duration sub-batches
    together, keeping the overall order roughly in book order.

    Params:
        items: (phase group index, retry_count) tuples, sorted by index
        voice_of_index: map from index to voice sample index (see
            effective_voice_indices)
        sort_by_words: False for CJK projects (no word-count sorting)
    """
    if not items:
        return []

    distinct_voices = {voice_of_index[index] for index, _ in items}
    window_size = batch_size * BATCH_ITERATIONS_PER_GROUP * len(distinct_voices)

    rounds: list[list[SubBatch]] = []
    is_descending = True

    for i in range(0, len(items), window_size):
        window = items[i:i + window_size]

        # Group the window's items by voice, in order of first appearance so
        # that a round's sub-batches stay roughly in book order.
        by_voice: dict[int, list[tuple[int, int]]] = {}
        voice_order: list[int] = []
        for item in window:
            voice = voice_of_index[item[0]]
            if voice not in by_voice:
                by_voice[voice] = []
                voice_order.append(voice)
            by_voice[voice].append(item)

        sub_batches: list[SubBatch] = []
        for voice in voice_order:
            group = by_voice[voice]
            if sort_by_words:
                group.sort(key=lambda item: phrase_groups[item[0]].num_words, reverse=is_descending)
            for j in range(0, len(group), batch_size):
                sub_batches.append(SubBatch(
                    voice_selection_index=voice,
                    items=tuple(group[j:j + batch_size]),
                ))
        rounds.append(sub_batches)
        is_descending = not is_descending

    return rounds

def make_retry_round(
        items: list[tuple[int, int]],
        voice_of_index: dict[int, int] | None,
        batch_size: int,
) -> list[SubBatch]:
    """
    Builds a single round from pending retries (re-added items). Items keep
    the order they were produced in and are grouped by voice sample (when
    `voice_of_index` is set); no word-count re-bucketing is
    done, matching the single-speaker behavior of processing re-adds promptly.
    """
    if voice_of_index is None:
        groups: list[tuple[int | None, list[tuple[int, int]]]] = [(None, list(items))]
    else:
        by_voice: dict[int, list[tuple[int, int]]] = {}
        for item in items:
            by_voice.setdefault(voice_of_index[item[0]], []).append(item)
        groups = [(voice, by_voice[voice]) for voice in sorted(by_voice)]

    sub_batches: list[SubBatch] = []
    for voice, group in groups:
        for i in range(0, len(group), batch_size):
            sub_batches.append(SubBatch(
                voice_selection_index=voice,
                items=tuple(group[i:i + batch_size]),
            ))
    return sub_batches
