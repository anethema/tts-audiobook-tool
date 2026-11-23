"""Find and merge duplicate characters in a browser book session."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import unicodedata

from tts_audiobook_tool.speaker_assign.roster_util import RosterUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.web_backend import (
    BookSession,
    WebBackendError,
    _load_analysis_checkpoint,
    _save_analysis_checkpoint,
)
from tts_audiobook_tool.web_render import RenderVoice


def suggest_duplicate_characters(session: BookSession | None) -> list[tuple[str, str]]:
    """Suggest pairs by shared names and small spelling variations."""
    if session is None or session.roster is None:
        return []

    characters = session.roster.characters
    narrator = session.roster.narrator
    narrator_id = narrator.id if narrator is not None else "NARRATOR"
    names = {character.id: _spelling_key(character.name) for character in characters}
    aliases = {
        character.id: {_spelling_key(alias) for alias in character.aliases}
        for character in characters
    }
    suggestions = []
    for left, right in combinations(characters, 2):
        left_name, right_name = names[left.id], names[right.id]
        if not left_name or not right_name:
            continue
        if (left_name in aliases[right.id] or right_name in aliases[left.id]
                or _similar_spelling(left_name, right_name)):
            # The first ID is the suggested keeper. The narrator can only be kept.
            pair = (right.id, left.id) if right.id == narrator_id else (left.id, right.id)
            suggestions.append(pair)
    return suggestions


def _spelling_key(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", value).casefold() if char.isalnum())


def _similar_spelling(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) < 5 or abs(len(left) - len(right)) > 1:
        return False

    differences = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
    if len(left) == len(right):
        if len(differences) == 1:
            return True
        if len(differences) == 2:
            first, second = differences
            return second == first + 1 and left[first:second + 1] == right[first:second + 1][::-1]
        return False

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    first = differences[0] if differences else len(shorter)
    return shorter[first:] == longer[first + 1:]


def merge_characters(
        session: BookSession,
        voices: Mapping[str, RenderVoice],
        keep_id: str,
        remove_id: str,
) -> tuple[BookSession, dict[str, RenderVoice]]:
    """Merge a confirmed pair and persist the roster and speaker labels together."""
    if keep_id == remove_id:
        raise WebBackendError("Choose two different characters to merge.")

    directory = session.attribution_dir or session.project_dir
    checkpoint = Path(SpeakerAssignFileUtil.attributions_path(directory))
    roster, attributions = _load_analysis_checkpoint(session, checkpoint)
    kept, removed = roster.get(keep_id), roster.get(remove_id)
    if kept is None or removed is None:
        raise WebBackendError("Choose two characters from the current cast.")
    narrator = roster.narrator
    if removed.id == (narrator.id if narrator is not None else "NARRATOR"):
        raise WebBackendError("Choose the narrator as the character to keep.")

    merged = replace(
        RosterUtil.merge_character(kept, removed),
        merged_ids=(*kept.merged_ids, removed.id, *removed.merged_ids),
    )
    updated_roster = replace(roster, characters=tuple(
        merged if character.id == keep_id else character
        for character in roster.characters if character.id != remove_id
    ))
    updated_attributions = replace(attributions, ranges=tuple(
        replace(attribution, utterances=tuple(
            replace(utterance, speaker=keep_id) if utterance.speaker == remove_id else utterance
            for utterance in attribution.utterances
        ))
        for attribution in attributions.ranges
    ))
    updated_voices = dict(voices)
    updated_voices.pop(remove_id, None)
    _save_analysis_checkpoint(checkpoint, updated_roster, updated_attributions)
    return replace(session, roster=updated_roster, attributions=updated_attributions), updated_voices
