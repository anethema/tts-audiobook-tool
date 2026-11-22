"""Review and correct saved dialogue assignments from the local browser."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace

from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_types import NOT_SPEECH, OTHER, UNKNOWN
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.web_backend import BookSession, WebBackendError


_RESERVED_NAMES = {
    OTHER: "Other speaker",
    UNKNOWN: "Unknown speaker",
    NOT_SPEECH: "Not spoken dialogue",
}


@dataclass(frozen=True)
class DialogueRow:
    first_line: int
    text: str
    speaker: str
    uncertain: bool


def dialogue_sections(session: BookSession | None) -> list[tuple[str, int]]:
    """List reviewable sections after the whole-book assignment has completed."""
    if session is None or session.roster is None or not session.attributions.ranges:
        return []
    plan = AssignPlan.make(session.book, session.attributions)
    if not plan.chunks or not plan.is_complete:
        return []

    assigned_lines = sorted(session.attributions.by_first_line(session.book))
    choices = []
    for index, (start, end) in enumerate(session.book.section_ranges()):
        first = bisect_left(assigned_lines, start)
        if first < len(assigned_lines) and assigned_lines[first] < end:
            choices.append((UtteranceUtil.section_label(session.book, index), index))
    return choices


def speaker_choices(session: BookSession) -> list[tuple[str, str]]:
    characters = session.roster.characters if session.roster else ()
    return [(character.name, character.id) for character in characters] + [
        (name, speaker) for speaker, name in _RESERVED_NAMES.items()
    ]


def dialogue_rows(session: BookSession, section: int) -> list[DialogueRow]:
    """Return complete quotation text and current speakers for one section."""
    start, end = _section_range(session, section)
    names = {speaker: name for name, speaker in speaker_choices(session)}
    assignments = session.attributions.by_first_line(session.book)
    return [
        DialogueRow(
            line,
            UtteranceUtil.range_text(session.book, line, utterance.lines[-1] + 1).strip(),
            names.get(utterance.speaker, utterance.speaker),
            utterance.uncertain,
        )
        for line, utterance in sorted(assignments.items())
        if start <= line < end
    ]


def correct_speaker(session: BookSession, section: int, first_line: int, speaker: str) -> BookSession:
    if section not in {index for _, index in dialogue_sections(session)}:
        raise WebBackendError("Complete the dialogue assignment before making corrections.")
    start, end = _section_range(session, section)
    if type(first_line) is not int or not start <= first_line < end:
        raise WebBackendError("Choose a dialogue line in the current section.")
    if speaker not in {speaker_id for _, speaker_id in speaker_choices(session)}:
        raise WebBackendError("Choose a speaker from the list.")

    original = next(
        (
            saved_range for saved_range in reversed(session.attributions.valid_ranges(session.book))
            if any(utterance.first_line == first_line for utterance in saved_range.utterances)
        ),
        None,
    )
    if original is None:
        raise WebBackendError("Choose a dialogue line with an assignment.")
    corrected = replace(
        original,
        utterances=tuple(
            replace(utterance, speaker=speaker, uncertain=False) if utterance.first_line == first_line else utterance
            for utterance in original.utterances
        ),
    )
    attributions = replace(
        session.attributions,
        ranges=tuple(
            corrected if saved_range.key == original.key else saved_range
            for saved_range in session.attributions.ranges
        ),
    )
    directory = session.attribution_dir or session.project_dir
    if SpeakerAssignFileUtil.save_attributions_file(directory, attributions):
        raise WebBackendError("Could not save the dialogue correction.")
    return replace(session, attributions=attributions)


def _section_range(session: BookSession, section: int) -> tuple[int, int]:
    ranges = session.book.section_ranges()
    if type(section) is not int or not 0 <= section < len(ranges):
        raise WebBackendError("Choose a book section.")
    return ranges[section]
