"""Review and correct saved dialogue assignments from the local browser."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from pathlib import Path

from tts_audiobook_tool.app_types.delivery_cue import DeliveryCue
from tts_audiobook_tool.speaker_assign.roster import (
    NEUTRAL_CHARACTER_VOICE_DIRECTION,
    is_current_character_voice_direction,
)
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_types import NOT_SPEECH, OTHER, UNKNOWN
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.web_backend import (
    BookSession, UNIDENTIFIED_CAST_ROLES, WebBackendError, _save_analysis_checkpoint, canonical_cast_role,
)


_RESERVED_NAMES = {
    **UNIDENTIFIED_CAST_ROLES,
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
    delivery: DeliveryCue | None = None
    voice_direction: str = ""

    @property
    def performance(self) -> str:
        if not self.delivery:
            return ""
        parts = [self.delivery.direction] if self.delivery.direction else []
        if self.delivery.before:
            parts.append(f"Before · {self.delivery.before}")
        if self.delivery.after:
            parts.append(f"After · {self.delivery.after}")
        return " / ".join(parts)


def dialogue_sections(session: BookSession | None) -> list[tuple[str, int]]:
    """List reviewable sections after the whole-book assignment has completed."""
    if session is None or session.roster is None or not session.attributions.ranges:
        return []
    assignments = session.attributions.by_first_line(session.book)
    covered = {line for assignment in assignments.values() for line in assignment.lines}
    dialogue = session.book.dialog_line_indices()
    if not dialogue or not covered.issuperset(dialogue):
        return []

    assigned_lines = sorted(assignments)
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


def display_voice_direction(direction: str) -> str:
    if direction != NEUTRAL_CHARACTER_VOICE_DIRECTION and is_current_character_voice_direction(direction):
        return direction
    return ""


def dialogue_rows(session: BookSession, section: int) -> list[DialogueRow]:
    """Return complete quotation text and current speakers for one section."""
    start, end = _section_range(session, section)
    names = {speaker: name for name, speaker in speaker_choices(session)}
    characters = session.roster.characters if session.roster else ()
    directions = {character.id: display_voice_direction(character.voice_direction) for character in characters}
    assignments = session.attributions.by_first_line(session.book)
    return [
        DialogueRow(
            line,
            UtteranceUtil.range_text(session.book, line, utterance.lines[-1] + 1).strip(),
            names.get(canonical_cast_role(utterance.speaker), utterance.speaker),
            utterance.uncertain,
            utterance.delivery,
            directions.get(utterance.speaker, ""),
        )
        for line, utterance in sorted(assignments.items())
        if start <= line < end
    ]


def correct_speaker(session: BookSession, section: int, first_line: int, speaker: str) -> BookSession:
    if session.roster is None or section not in {index for _, index in dialogue_sections(session)}:
        raise WebBackendError("Complete the dialogue assignment before making corrections.")
    start, end = _section_range(session, section)
    if type(first_line) is not int or not start <= first_line < end:
        raise WebBackendError("Choose a dialogue line in the current section.")
    if isinstance(speaker, str):
        speaker = canonical_cast_role(speaker)
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
    checkpoint = Path(SpeakerAssignFileUtil.attributions_path(directory))
    _save_analysis_checkpoint(checkpoint, session.roster, attributions)
    return replace(session, attributions=attributions)


def _section_range(session: BookSession, section: int) -> tuple[int, int]:
    ranges = session.book.section_ranges()
    if type(section) is not int or not 0 <= section < len(ranges):
        raise WebBackendError("Choose a book section.")
    return ranges[section]
