"""EPUB and speaker assignment operations for the local browser interface."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import groupby
import json
from pathlib import Path
import re
import shutil
import tempfile
from time import monotonic
from urllib.parse import urlsplit
from typing import TYPE_CHECKING

import requests

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.app_types.delivery_cue import DeliveryCue
from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_DEFAULT
from tts_audiobook_tool.constants_config import PROJECT_DEFAULT_SEGMENTATION_STRATEGY
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan, PlannedChunk
from tts_audiobook_tool.speaker_assign.attribution import (
    UTTERANCE_TEXT_EXCERPT_CHARS, AttributionsFile, RangeAttribution, UtteranceAttribution,
)
from tts_audiobook_tool.speaker_assign.attribution_prompt_util import ATTRIBUTION_SYSTEM_PROMPT
from tts_audiobook_tool.speaker_assign.attribution_reply_util import AttributionReplyUtil
from tts_audiobook_tool.speaker_assign.combined_analysis import SYSTEM_PROMPT as COMBINED_ANALYSIS_SYSTEM_PROMPT
from tts_audiobook_tool.speaker_assign.combined_analysis import (
    CHARACTER_VOICE_SYSTEM_INSTRUCTIONS, PERFORMANCE_CUE_SYSTEM_PROMPT,
    InvalidChunkReplyError, build_prompt, parse_chunk_reply,
)
from tts_audiobook_tool.speaker_assign.attribution_runner import AttributionRunner, AttributionRunListener
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult, SPEAKER_ASSIGN_LLM_TIMEOUT
from tts_audiobook_tool.speaker_assign.roster import Roster, is_current_character_voice_direction
from tts_audiobook_tool.speaker_assign.roster_util import ROSTER_SYSTEM_PROMPT, RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_types import NOT_SPEECH, Utterance
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor
from tts_audiobook_tool.web_local_tagger import LOCAL_QWEN_MODEL, LocalQwen, LocalTaggerError

if TYPE_CHECKING:
    from tts_audiobook_tool.web_render import RenderVoice


DEFAULT_OPENAI_MODEL = "gpt-6.1-terra"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_LLM_ENDPOINT = "https://api.openai.com/v1/chat/completions"
UNIDENTIFIED_CAST_ROLES: dict[str, str] = {
    "OTHER:unidentified female": "Unidentified female",
    "OTHER:unidentified male": "Unidentified male",
}
_WEB_ATTRIBUTION_SYSTEM_PROMPT = ATTRIBUTION_SYSTEM_PROMPT + """

Use a named roster character's ID whenever you can identify the speaker. If
the speaker is unnamed or their identity is unresolved and the text indicates
their gender, use exactly "OTHER:unidentified female" or "OTHER:unidentified male"
as appropriate. Use "UNKNOWN" when the evidence is insufficient to identify
the speaker or place them in either group. Use "NOT_SPEECH" for quoted text
that is not spoken dialogue.
"""
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_FINISH_REASONS = ("stop", "length", "content_filter", "tool_calls", "function_call")
_ANALYSIS_REPLY_RETRIES = 2
_WORKSPACE = tempfile.TemporaryDirectory(prefix="tts-audiobook-web-", ignore_cleanup_errors=True)


class WebBackendError(ValueError):
    """An error whose message can be shown in the browser."""


@dataclass(frozen=True, eq=False)
class BookSession:
    """A book revision with identity-based change tracking in the browser."""

    book: Book
    project_dir: str
    warnings: tuple[str, ...] = ()
    roster: Roster | None = None
    attributions: AttributionsFile = field(default_factory=AttributionsFile)
    attribution_notice: str = ""
    attribution_dir: str | None = None
    project_language: str | None = None
    word_substitutions: dict[str, str] = field(default_factory=dict)
    line_voices: dict[int, RenderVoice] = field(default_factory=dict)

    @property
    def language_code(self) -> str:
        if self.project_language is not None:
            return self.project_language
        return self.book.segmentation_settings.language_code or "en"

    def summary(self) -> str:
        return (
            f"{len(self.book.sections):,} sections · {len(self.book.phrase_groups):,} lines"
            f" · {len(self.book.dialog_line_indices()):,} dialogue lines"
        )


@dataclass(frozen=True)
class CharacterRow:
    speaker_id: str
    name: str
    line_count: int
    voice_hint: str
    voice_direction: str = ""
    gender: str = "unknown"


def canonical_cast_role(speaker: str) -> str:
    """Normalize the two browser casting groups while preserving other speaker IDs."""
    prefix, separator, name = speaker.partition(":")
    if separator and prefix.strip().casefold() == "other":
        normalized = " ".join(name.casefold().replace("_", " ").split())
        role = f"OTHER:{normalized}"
        if role in UNIDENTIFIED_CAST_ROLES:
            return role
    return speaker


def import_epub(upload_path: str, *, language_code: str = "en") -> BookSession:
    path = Path(upload_path or "")
    if path.suffix.lower() != ".epub" or not path.is_file():
        raise WebBackendError("Choose an EPUB file to import.")
    language_code = language_code.strip() or "en"

    try:
        imported = EpubExtractor.import_epub(
            str(path), MAX_WORDS_PER_SEGMENT_DEFAULT, PROJECT_DEFAULT_SEGMENTATION_STRATEGY,
            language_code, dialog_segmentation=True,
        )
    except Exception:
        raise WebBackendError("The EPUB could not be read. Check that the file opens in an EPUB reader.") from None
    if not imported.phrase_groups:
        raise WebBackendError("The EPUB contains no readable text.")

    book = ProjectTextIOUtil.make_import_book(
        imported.phrase_groups, PROJECT_DEFAULT_SEGMENTATION_STRATEGY, MAX_WORDS_PER_SEGMENT_DEFAULT,
        language_code, dialog_segmentation=True, title=imported.book_title.strip() or path.stem,
        text_source_kind="epub", section_start_indices=imported.section_start_indices,
        section_titles=[chapter.title for chapter in imported.chapters],
    )
    return new_book_session(book, warnings=tuple(imported.significant_warnings))


def new_book_session(
        book: Book,
        *,
        project_dir: str | None = None,
        warnings: tuple[str, ...] = (),
        roster: Roster | None = None,
        attributions: AttributionsFile | None = None,
        project_language: str | None = None,
        word_substitutions: dict[str, str] | None = None,
        line_voices: dict[int, RenderVoice] | None = None,
) -> BookSession:
    """Create an isolated analysis workspace, optionally opening a saved book."""
    workspace = tempfile.mkdtemp(dir=_WORKSPACE.name)
    loaded = attributions if attributions is not None else AttributionsFile()
    session = BookSession(
        book=book,
        project_dir=project_dir or workspace,
        warnings=warnings,
        roster=roster,
        attributions=loaded,
        attribution_dir=workspace,
        project_language=project_language,
        word_substitutions=dict(word_substitutions or {}),
        line_voices=dict(line_voices or {}),
    )
    if loaded.ranges and SpeakerAssignFileUtil.save_attributions_file(workspace, loaded):
        close_book_session(session)
        raise WebBackendError("Saved dialogue assignments could not be copied into the session workspace.")
    return session


def close_book_session(session: BookSession | None) -> None:
    """Clear the temporary workspace for a book owned by this process."""
    if session is None:
        return
    project_dir = Path(session.attribution_dir or session.project_dir)
    if project_dir.resolve().parent == Path(_WORKSPACE.name).resolve():
        shutil.rmtree(project_dir, ignore_errors=True)


def character_rows(session: BookSession) -> list[CharacterRow]:
    roster = session.roster
    narrator = roster.narrator if roster else None
    narrator_id = narrator.id if narrator else "NARRATOR"
    characters = roster.characters if roster else ()
    counts = Counter(session.attributions.speaker_of_each_line(session.book).values())
    lines_for_other_characters = sum(counts[character.id] for character in characters if character.id != narrator_id)
    narrator_hint = narrator.voice_hint if narrator else ""
    if not narrator_hint and roster is not None:
        narrator_hint = roster.narration.voice_hint

    return [
        CharacterRow(
            narrator_id,
            narrator.name if narrator else "Narrator",
            len(session.book.phrase_groups) - lines_for_other_characters,
            narrator_hint,
            voice_direction=narrator.voice_direction if narrator else "",
            gender=narrator.gender if narrator else "unknown",
        ),
        *(
            CharacterRow(
                character.id, character.name, counts[character.id], character.voice_hint,
                voice_direction=character.voice_direction, gender=character.gender,
            )
            for character in characters if not character.is_narrator and character.id != narrator_id
        ),
    ]


def book_cast_problem(
    session: BookSession, *, first_line: int = 1, last_line: int | None = None,
) -> str | None:
    """Check unattributed dialogue in an inclusive, one-based render range."""
    if session.roster is None:
        return None
    groups = session.book.phrase_groups
    last_line = len(groups) if last_line is None else last_line
    if type(first_line) is not int or type(last_line) is not int or not 1 <= first_line <= last_line <= len(groups):
        return None

    speakers = session.attributions.speaker_of_each_line(session.book)
    if any(
        groups[index].is_dialog and index not in speakers
        and not (groups[index].voice_index > 0 and groups[index].voice_index in session.line_voices)
        for index in range(first_line - 1, last_line)
    ):
        return "Run Analyze book for the selected lines before generating audio."
    return None


def analyze_book_dialogue(
        session: BookSession,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        effort: str = DEFAULT_REASONING_EFFORT,
        endpoint: str = DEFAULT_LLM_ENDPOINT,
        settings_json: str = "",
        local_tagger: LocalQwen | None = None,
        performance_cues: bool = False,
        character_voices: bool = False,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
        on_work_progress: Callable[[int, int], None] | None = None,
) -> BookSession:
    """Report overall chunk progress and the requested work completed during this run."""
    log = on_log if on_log is not None else lambda _message: None
    directory = session.attribution_dir or session.project_dir
    checkpoint = Path(SpeakerAssignFileUtil.attributions_path(directory))
    roster, attributions = _load_analysis_checkpoint(session, checkpoint)
    book = session.book
    plan = AssignPlan.make(book, attributions)
    total = len(plan.chunks)
    if not total:
        raise WebBackendError("This book has no dialogue lines to analyze.")
    effective = attributions.by_first_line(book)
    prior_speakers = {line: utterance.speaker for line, utterance in effective.items()}
    effective_by_line = {line: stored for stored in effective.values() for line in stored.lines}
    utterances_by_chunk = {
        chunk.number: UtteranceUtil.find_utterances(book, chunk.start, chunk.end) for chunk in plan.chunks
    }
    speakers_by_chunk = {
        number: {
            quote_id for quote_id, utterance in enumerate(utterances, 1)
            if any(line not in effective_by_line for line in utterance.lines)
        }
        for number, utterances in utterances_by_chunk.items()
    }
    cues_by_chunk: dict[int, set[int]] = {}
    if performance_cues:
        whole_quotes = {
            utterance.lines for utterance in UtteranceUtil.find_utterances(book, 0, len(book.flat_phrase_groups))
        }
        cues_by_chunk = {
            number: {
                quote_id for quote_id, utterance in enumerate(utterances, 1)
                if utterance.lines in whole_quotes and _needs_delivery(utterance, effective_by_line)
            }
            for number, utterances in utterances_by_chunk.items()
        }
    voices_by_chunk = _character_voice_requests(roster, utterances_by_chunk, effective_by_line) if character_voices else {}
    work_by_chunk = {
        number: len(speakers | cues_by_chunk.get(number, set())) + len(voices_by_chunk.get(number, ()))
        for number, speakers in speakers_by_chunk.items()
    }
    complete = sum(units == 0 for units in work_by_chunk.values())
    completed_work = 0
    total_work = sum(work_by_chunk.values())

    def progress(detail: str) -> None:
        if on_work_progress is not None:
            on_work_progress(completed_work, total_work)
        if on_progress is not None:
            on_progress(complete, total, detail)

    if complete:
        progress(f"Reusing saved analysis for {_count_label(complete, 'chunk')}.")
    else:
        progress("Starting dialogue analysis.")
    prefs = None
    analysis_model = LOCAL_QWEN_MODEL if local_tagger is not None else model.strip()
    issue = ""
    skipped_chunks: list[PlannedChunk] = []
    log(f"Analyzing {total:,} dialogue chunks.\n")
    for chunk in plan.chunks:
        label = f"Chunk {chunk.number:,}/{total:,}"
        utterances = utterances_by_chunk[chunk.number]
        requested = speakers_by_chunk[chunk.number]
        cue_requests = cues_by_chunk.get(chunk.number, set())
        voice_requests = voices_by_chunk.get(chunk.number, set())
        if not requested and not cue_requests and not voice_requests:
            quotes = _count_label(len(utterances), "quote")
            saved = "analysis" if performance_cues else "speaker labels"
            log(f"{label} reused saved {saved} for {quotes}.\n")
            continue
        line_range = f"lines {chunk.start + 1:,}–{chunk.end:,}"
        location = f"{label} ({line_range})"
        try:
            if local_tagger is None and prefs is None:
                api_key, prefs = _request_settings(api_key, model, effort, endpoint, settings_json)
            saved_speakers = {
                number: next(iter(speakers))
                for number, utterance in enumerate(utterances, 1) if number not in requested
                if len(speakers := {effective_by_line[line].speaker for line in utterance.lines}) == 1
            }
            prompt = build_prompt(
                book, chunk, roster, prior_speakers, saved_speakers,
                speaker_requests=requested, cue_requests=cue_requests, voice_requests=voice_requests,
            )
            request_parts = []
            if requested:
                request_parts.append(f"speakers for {_count_label(len(requested), 'quote')}")
            if cue_requests:
                request_parts.append(f"delivery cues for {_count_label(len(cue_requests), 'quote')}")
            if voice_requests:
                request_parts.append(f"voice directions for {_count_label(len(voice_requests), 'character')}")
            request_description = " and ".join(request_parts)
            progress(f"{location} · requesting {request_description}")
            log(f"{label} requesting {request_description} ({line_range}).\n")
            system_prompt = PERFORMANCE_CUE_SYSTEM_PROMPT if cue_requests else COMBINED_ANALYSIS_SYSTEM_PROMPT
            if character_voices:
                system_prompt += CHARACTER_VOICE_SYSTEM_INSTRUCTIONS
            attempt_system_prompt = system_prompt
            for attempt in range(_ANALYSIS_REPLY_RETRIES + 1):
                if local_tagger is None:
                    reply = _send(_make_session(prefs, api_key, attempt_system_prompt), prompt)
                else:
                    reply = local_tagger.send(attempt_system_prompt, prompt)
                if reply.error is not None:
                    raise WebBackendError(_request_error_message(reply.error))
                if reply.hit_output_limit:
                    raise WebBackendError("The model reached its output limit. Try lower reasoning effort or a higher output limit.")
                try:
                    next_roster, parsed = parse_chunk_reply(
                        reply.reply, roster, requested, cue_requests=cue_requests,
                        voice_requests=voice_requests, character_voices=character_voices,
                    )
                except (ValueError, RecursionError) as error:
                    if isinstance(error, InvalidChunkReplyError):
                        reason = str(error)
                    elif isinstance(error, json.JSONDecodeError):
                        reason = "The reply could not be parsed as JSON."
                    else:
                        reason = "The reply did not match the expected analysis format."
                    if attempt < _ANALYSIS_REPLY_RETRIES:
                        progress(f"{location} · retry {attempt + 1}/{_ANALYSIS_REPLY_RETRIES} after invalid reply")
                        log(f"{label} {reason} Retrying ({attempt + 1}/{_ANALYSIS_REPLY_RETRIES}).\n")
                        attempt_system_prompt = (
                            f"{system_prompt}\n\nRetry reminder: {reason} Return one valid JSON object. "
                            "Include exactly the quote IDs needing speaker labels in speakers and use valid speaker IDs."
                        )
                else:
                    break
            else:
                skipped_chunks.append(chunk)
                progress(f"{location} · skipped after {attempt + 1} invalid replies")
                log(f"{label} skipped ({line_range}) after {attempt + 1} invalid replies. {reason}\n")
                continue

            next_attributions, new_utterances = attributions, []
            if requested or cue_requests:
                added = AttributionReplyUtil.make_range_attribution(
                    book, chunk.start, chunk.end, utterances, parsed, llm={"model": analysis_model},
                )
                next_attributions, new_utterances = _merge_chunk_analysis(
                    book, attributions, added, requested, effective_by_line,
                )
            if next_roster != roster or next_attributions != attributions:
                _save_analysis_checkpoint(checkpoint, next_roster, next_attributions)
        except (LocalTaggerError, WebBackendError) as exc:
            issue = str(exc)
            progress(f"{location} · stopped")
            log(f"{label} stopped. {issue} Run Analyze book to retry this chunk.\n")
            break
        expected_voices = voice_requests | (next_roster.ids - roster.ids) if character_voices else set()
        added_voices = sum(
            is_current_character_voice_direction(entry.voice_direction)
            for entry in next_roster.characters if entry.id in expected_voices
        )
        missing_voices = len(expected_voices) - added_voices
        roster, attributions = next_roster, next_attributions
        prior_speakers.update((utterance.first_line, utterance.speaker) for utterance in new_utterances)
        if missing_voices:
            progress(f"{location} · missing {_count_label(missing_voices, 'character voice direction')}")
        else:
            complete += 1
            completed_work += work_by_chunk[chunk.number]
            progress(f"{location} · complete")
        usage = ""
        if reply.input_tokens is not None and reply.output_tokens is not None:
            usage = f" {reply.input_tokens:,} input / {reply.output_tokens:,} output tokens."
        saved_parts = []
        if requested:
            saved_parts.append(_count_label(len(requested), "speaker label"))
        if cue_requests:
            saved_parts.append(_count_label(sum(bool(cue) for cue in parsed.delivery.values()), "delivery cue"))
        if added_voices:
            saved_parts.append(_count_label(added_voices, "character voice direction"))
        if saved_parts:
            log(f"{label} saved {' and '.join(saved_parts)} in {reply.elapsed:.1f}s.{usage}\n")
        if missing_voices:
            log(f"{label} did not receive {_count_label(missing_voices, 'usable character voice direction')}.\n")

    if issue and not attributions.ranges and not skipped_chunks:
        raise WebBackendError(issue)
    notice = f"{complete:,} of {total:,} dialogue chunks complete."
    if skipped_chunks:
        chunks = "chunk" if len(skipped_chunks) == 1 else "chunks"
        locations = ", ".join(
            f"{chunk.number:,} (lines {chunk.start + 1:,}–{chunk.end:,})" for chunk in skipped_chunks
        )
        notice += f" Skipped {chunks} {locations} after invalid replies."
    remaining_voices = sum(
        not is_current_character_voice_direction(entry.voice_direction) for entry in roster.characters
    ) if character_voices else 0
    if remaining_voices:
        notice += f" Missing {_count_label(remaining_voices, 'character voice direction')}."
    if issue:
        notice += f" {issue}"
    if skipped_chunks:
        notice += " Run Analyze book to retry incomplete chunks."
    elif issue or remaining_voices:
        notice += " Run Analyze book to continue."
    log(notice + "\n")
    return replace(session, roster=roster, attributions=attributions, attribution_notice=notice)


def _character_voice_requests(
        roster: Roster, utterances_by_chunk: dict[int, list[Utterance]],
        saved_by_line: dict[int, UtteranceAttribution],
) -> dict[int, set[str]]:
    """Request each missing direction once, where its saved dialogue first appears."""
    remaining = {entry.id for entry in roster.characters if not is_current_character_voice_direction(entry.voice_direction)}
    redirects = {old_id: entry.id for entry in roster.characters for old_id in entry.merged_ids}
    requests_by_chunk = {}
    for number, utterances in utterances_by_chunk.items():
        if not remaining:
            break
        speakers = {
            redirects.get(saved_by_line[line].speaker, saved_by_line[line].speaker)
            for utterance in utterances for line in utterance.lines if line in saved_by_line
        }
        requested = remaining & speakers
        if requested:
            requests_by_chunk[number] = requested
            remaining.difference_update(requested)
    if remaining:
        # A roster can include characters whose dialogue was edited or removed.
        # Their roster descriptions still supply evidence in this bounded request.
        requests_by_chunk.setdefault(next(iter(utterances_by_chunk)), set()).update(remaining)
    return requests_by_chunk


def _needs_delivery(utterance: Utterance, saved_by_line: dict[int, UtteranceAttribution]) -> bool:
    """Only attach delivery where the entire quote has a single exact identity."""
    first = saved_by_line.get(utterance.first_line)
    if first is None:
        return all(line not in saved_by_line for line in utterance.lines)
    return (
        first.lines == utterance.lines and first.delivery is None and first.speaker != NOT_SPEECH
        and all(saved_by_line.get(line) is first for line in utterance.lines)
    )


def _merge_chunk_analysis(
        book: Book,
        attributions: AttributionsFile,
        candidate: RangeAttribution,
        requested: set[int],
        saved_by_line: dict[int, UtteranceAttribution],
) -> tuple[AttributionsFile, list[UtteranceAttribution]]:
    """Enrich effective saved quotes in place and add labels for uncovered lines."""
    delivery_by_line: dict[int, DeliveryCue] = {}
    new_utterances: list[UtteranceAttribution] = []
    for number, utterance in enumerate(candidate.utterances, 1):
        if number not in requested:
            if utterance.delivery is not None:
                delivery_by_line[utterance.first_line] = utterance.delivery
            continue
        for assigned, members in groupby(utterance.lines, key=saved_by_line.__contains__):
            if assigned:
                continue
            lines = tuple(members)
            if lines == utterance.lines:
                new_utterances.append(utterance)
            else:
                text = UtteranceUtil.range_text(book, lines[0], lines[-1] + 1).strip()
                new_utterances.append(replace(
                    utterance, lines=lines, text=text[:UTTERANCE_TEXT_EXCERPT_CHARS], delivery=None,
                ))

    ranges = list(attributions.ranges)
    for index in range(len(ranges) - 1, -1, -1):
        if not delivery_by_line:
            break
        saved = ranges[index]
        updated = tuple(
            replace(utterance, delivery=delivery_by_line.pop(utterance.first_line))
            if utterance.first_line in delivery_by_line and saved_by_line[utterance.first_line] is utterance
            else utterance
            for utterance in saved.utterances
        )
        if updated != saved.utterances:
            ranges[index] = replace(saved, utterances=updated)

    if new_utterances:
        for index, saved in enumerate(ranges):
            if saved.key == candidate.key:
                updated = tuple(sorted((*saved.utterances, *new_utterances), key=lambda quote: quote.first_line))
                ranges[index] = replace(saved, utterances=updated)
                break
        else:
            ranges.append(replace(candidate, utterances=tuple(new_utterances)))
    return AttributionsFile(tuple(ranges)), new_utterances


def _count_label(count: int, singular: str) -> str:
    return f"{count:,} {singular}{'' if count == 1 else 's'}"


def _load_analysis_checkpoint(session: BookSession, checkpoint: Path) -> tuple[Roster, AttributionsFile]:
    roster = session.roster or Roster(())
    try:
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return roster, AttributionsFile(tuple(session.attributions.valid_ranges(session.book)))
    except (OSError, ValueError):
        raise WebBackendError("Saved dialogue assignments could not be read.") from None
    try:
        saved = AttributionsFile(tuple(AttributionsFile.from_dict(value).valid_ranges(session.book)))
        if "combined_roster" in value:
            roster = Roster.from_dict(value["combined_roster"])
    except ValueError:
        raise WebBackendError("Saved dialogue assignments could not be read.") from None
    return roster, saved


def _save_analysis_checkpoint(checkpoint: Path, roster: Roster, attributions: AttributionsFile) -> None:
    """Store roster and quote labels in one write, using the canonical attribution file."""
    try:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise WebBackendError("Dialogue analysis progress could not be saved.") from None
    if JsonSaveUtil.save(
        JsonArtifactType.SPEAKER_ATTRIBUTIONS, checkpoint,
        lambda: {**attributions.to_dict(), "combined_roster": roster.to_dict()},
    ):
        raise WebBackendError("Dialogue analysis progress could not be saved.")


def identify_characters(
        session: BookSession,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        effort: str = DEFAULT_REASONING_EFFORT,
        endpoint: str = DEFAULT_LLM_ENDPOINT,
        settings_json: str = "",
) -> BookSession:
    api_key, prefs = _request_settings(api_key, model, effort, endpoint, settings_json)
    if not session.book.dialog_line_indices():
        raise WebBackendError("This book has no dialogue lines to identify.")

    llm = _make_session(prefs, api_key, ROSTER_SYSTEM_PROMPT)
    result = _send(llm, RosterUtil.build_message(session.book))
    if result.error is not None:
        raise WebBackendError(_request_error_message(result.error))
    if result.hit_output_limit:
        raise WebBackendError("The model reached its output limit. Try lower reasoning effort or a shorter book.")
    try:
        roster = RosterUtil.normalize_roster(AttributionReplyUtil.extract_json(result.reply)).roster
    except ValueError:
        raise WebBackendError("The model did not return a readable character roster. Try again.") from None
    if SpeakerAssignFileUtil.delete_attributions_file(session.attribution_dir or session.project_dir):
        raise WebBackendError("The previous dialogue assignments could not be cleared from the session workspace.")
    return replace(session, roster=roster, attributions=AttributionsFile(), attribution_notice="")


def attribute_dialog(
        session: BookSession,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        effort: str = DEFAULT_REASONING_EFFORT,
        endpoint: str = DEFAULT_LLM_ENDPOINT,
        settings_json: str = "",
        on_progress: Callable[[int, int], None] | None = None,
) -> BookSession:
    if session.roster is None:
        raise WebBackendError("Identify the characters before attributing dialogue.")
    api_key, prefs = _request_settings(api_key, model, effort, endpoint, settings_json)

    attribution_dir = session.attribution_dir or session.project_dir
    attributions = _read_attributions(attribution_dir)
    plan = AssignPlan.make(session.book, attributions)
    progress = _ProgressReporter(on_progress, plan)
    progress.report()
    last_request: LlmRequestResult | None = None

    def send(llm: LlmSession, prompt: str) -> LlmRequestResult:
        nonlocal last_request
        last_request = _send(llm, prompt)
        return last_request

    runner = AttributionRunner(
        book=session.book,
        roster=session.roster,
        attributions=attributions,
        project_dir=attribution_dir,
        prefs=prefs,
        make_session=lambda: _make_session(prefs, api_key, _WEB_ATTRIBUTION_SYSTEM_PROMPT),
        send=send,
        listener=progress,
    )
    result = runner.run()
    saved_attributions = _read_attributions(attribution_dir)
    complete = AssignPlan.make(session.book, saved_attributions).done_count
    total = len(plan.chunks)
    if complete < total:
        if last_request is not None and last_request.error is not None:
            notice = _request_error_message(last_request.error)
        elif last_request is not None and last_request.hit_output_limit:
            notice = "OpenAI reached its output limit. Try lower reasoning effort."
        elif result.error or result.failures:
            notice = "One or more dialogue batches could not be completed. Check the model and reasoning effort and retry."
        else:
            notice = "Some dialogue remains to be attributed. Run attribution again to continue."
        if not complete:
            raise WebBackendError(notice)
        notice = f"{complete:,} of {total:,} dialogue batches complete. {notice}"
    else:
        notice = f"{complete:,} dialogue batches complete."
    return replace(session, attributions=saved_attributions, attribution_notice=notice)


class _ProgressReporter(AttributionRunListener):
    def __init__(self, callback: Callable[[int, int], None] | None, plan: AssignPlan) -> None:
        self.callback = callback
        self.completed = plan.done_count
        self.total = len(plan.chunks)

    def report(self) -> None:
        if self.callback is not None:
            self.callback(self.completed, self.total)

    def chunk_saved(self, chunk: PlannedChunk, attribution: RangeAttribution, elapsed: float) -> None:
        self.completed += 1
        self.report()


def _read_attributions(project_dir: str) -> AttributionsFile:
    saved = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    if isinstance(saved, str):
        raise WebBackendError("Dialogue assignments could not be read from the session workspace.")
    return saved


def _request_settings(api_key: str, model: str, effort: str, endpoint: str, settings_json: str) -> tuple[str, Prefs]:
    endpoint = (endpoint or "").strip()
    try:
        address = urlsplit(endpoint)
        if (address.scheme not in ("http", "https") or not address.hostname
                or address.username is not None or address.fragment
                or (address.hostname == "api.openai.com" and address.scheme != "https")):
            raise ValueError
    except ValueError:
        raise WebBackendError("Enter a full HTTP or HTTPS chat completions endpoint URL.") from None

    api_key = api_key.strip()
    if (address.hostname == "api.openai.com" or api_key) and not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", api_key):
        raise WebBackendError("Paste a valid OpenAI API key.")
    model = model.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", model) or (api_key and api_key in model):
        raise WebBackendError("Enter a model ID.")
    effort = effort.strip()
    if effort and effort not in _REASONING_EFFORTS:
        raise WebBackendError("Choose a supported reasoning effort.")
    try:
        extra_params = json.loads(settings_json.strip() or "{}")
        if not isinstance(extra_params, dict):
            raise ValueError
    except ValueError:
        raise WebBackendError("Enter additional request parameters as a JSON object.") from None
    if extra_params.keys() & {"messages", "model", "store", "stream"}:
        raise WebBackendError("The app sets messages, model, store and stream for character analysis.")
    if effort:
        extra_params.setdefault("reasoning_effort", effort)
    return api_key, Prefs(
        llm_url=endpoint,
        llm_model=model,
        llm_speaker_assign_extra_params=extra_params,
    )


def _make_session(prefs: Prefs, api_key: str, system_prompt: str) -> LlmSession:
    return _WebOpenAISession(
        api_endpoint_url=prefs.llm_url,
        token=api_key,
        model=prefs.llm_model,
        api_type="openai",
        system_prompt=system_prompt,
        extra_params=prefs.llm_speaker_assign_extra_params,
        timeout=SPEAKER_ASSIGN_LLM_TIMEOUT,
    )


def _send(llm: LlmSession, prompt: str) -> LlmRequestResult:
    started = monotonic()
    try:
        reply = llm.send(prompt)
    except (requests.RequestException, WebBackendError) as error:
        return LlmRequestResult("", "", monotonic() - started, error=error)
    return LlmRequestResult(
        reply, "", monotonic() - started, finish_reason=llm.last_finish_reason,
        input_tokens=llm.last_input_tokens, output_tokens=llm.last_output_tokens,
        reasoning_tokens=llm.last_reasoning_tokens,
    )


class _SanitizedHttpError(requests.HTTPError):
    """A provider status with no response text, request headers or raw JSON."""

    def __init__(self, status: int, code: str = "") -> None:
        response = requests.Response()
        response.status_code = status
        super().__init__(f"OpenAI request failed with HTTP {status}.", response=response)
        self.code = code


def _request_error_message(error: Exception) -> str:
    if isinstance(error, _SanitizedHttpError):
        if error.code == "context_length_exceeded":
            return "This text exceeds the OpenAI model’s context limit. Try a model with a larger context window."
        if error.code == "insufficient_quota":
            return "OpenAI reported insufficient API quota. Check billing and usage for the API key."
        if error.response is not None:
            status = error.response.status_code
            if status in (401, 403):
                return "OpenAI denied the request. Check the API key and its project permissions."
            if status == 429:
                return "OpenAI rate-limited the request. Wait briefly and retry."
            if status in (400, 404):
                return f"OpenAI rejected the request (HTTP {status}). Check the model and reasoning effort."
        return str(error)
    if isinstance(error, requests.Timeout):
        return "The OpenAI request timed out. Try again."
    if isinstance(error, requests.ConnectionError):
        return "Could not connect to OpenAI. Check the network connection and retry."
    return "OpenAI did not return a usable answer. Try again."


class _WebOpenAISession(LlmSession):
    """Use the upstream chat state with a transport scoped to a pasted key."""

    def call_openai(self, payload, on_chunk, interrupt_event=None, on_reasoning=None) -> str:
        if "temperature" not in self.extra_params:
            payload.pop("temperature", None)
        if urlsplit(self.api_endpoint_url).hostname == "api.openai.com":
            payload["store"] = False
        if self.token and any(self.token in message["content"] for message in payload["messages"]):
            raise WebBackendError("The request text contains the API key. Remove it from the text before retrying.")
        try:
            with requests.Session() as http:
                # Disable .netrc auth overrides and ambient proxy configuration.
                http.trust_env = False
                with http.post(
                    self.api_endpoint_url,
                    headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
                    json=payload,
                    timeout=self.timeout,
                    allow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        raise _SanitizedHttpError(response.status_code, _safe_error_code(response))
                    body = response.json()
                    choice = body["choices"][0]
                    content = choice["message"]["content"]
                    self._record_usage(body.get("usage"), "completion_tokens", "prompt_tokens")
                    finish_reason = choice.get("finish_reason")
                    self.last_finish_reason = finish_reason if finish_reason in _FINISH_REASONS else None
        except _SanitizedHttpError:
            raise
        except requests.Timeout:
            raise requests.Timeout("OpenAI request timed out.") from None
        except requests.RequestException:
            raise requests.ConnectionError("Could not connect to OpenAI.") from None
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            raise WebBackendError("OpenAI returned an invalid response.") from None
        if content is None:
            return ""
        if not isinstance(content, str) or (self.token and self.token in content):
            raise WebBackendError("OpenAI returned an invalid response.")
        return content


def _safe_error_code(response: requests.Response) -> str:
    try:
        code = response.json()["error"]["code"]
    except (KeyError, TypeError, ValueError):
        return ""
    return code if code in ("context_length_exceeded", "insufficient_quota") else ""
