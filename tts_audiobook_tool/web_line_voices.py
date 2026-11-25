"""Manual voice overrides for browser audiobook lines."""
from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.web_backend import BookSession, WebBackendError

if TYPE_CHECKING:
    from tts_audiobook_tool.web_render import RenderVoice


LINE_VOICE_PAGE_SIZE = 25


def set_line_voice(
        session: BookSession | None, ranges: str, voice: RenderVoice | None,
) -> tuple[BookSession, int]:
    """Assign a saved voice snapshot to selected lines, or clear their overrides."""
    if session is None:
        raise WebBackendError("Open a book before assigning line voices.")
    lines = session.book.phrase_groups
    if not lines:
        raise WebBackendError("This book contains no lines to assign.")
    selected = _parse_ranges(ranges, len(lines))
    voices = dict(session.line_voices)
    slot = 0
    if voice is not None:
        slot = next((index for index, saved in voices.items() if index > 0 and saved == voice), 0)
        if not slot:
            slot = max(voices, default=0) + 1
            in_use = {line.voice_index for line in lines}
            while slot in in_use:
                slot += 1
            voices[slot] = voice

    sections = list(session.book.sections)
    changes = 0
    for index, (start, end) in enumerate(session.book.section_ranges()):
        section = sections[index]
        replacements = list(section.phrase_groups)
        changed = False
        for position in range(start, end):
            group = lines[position]
            if position not in selected or group.voice_index == slot:
                continue
            replacement = copy(group)
            replacement.voice_index = slot
            replacements[position - start] = replacement
            changes += 1
            changed = True
        if changed:
            sections[index] = replace(section, phrase_groups=replacements)

    if not changes:
        return session, 0
    return replace(session, book=replace(session.book, sections=sections), line_voices=voices), changes


def _parse_ranges(ranges: str, count: int) -> set[int]:
    if not isinstance(ranges, str) or not ranges.strip():
        raise WebBackendError("Enter line numbers or an explicit range such as 4-7.")
    try:
        selected, warnings = RangeStringUtil.parse_ranges_string(ranges, count, strict=True)
    except ValueError:
        raise WebBackendError("Use line numbers from 1 to the last line in the book.") from None
    if warnings:
        raise WebBackendError(f"{warnings[0]}. Use line numbers from 1 to {count:,}.")
    if not selected:
        raise WebBackendError("Select at least one line.")
    return selected


def line_voice_sections(session: BookSession | None) -> list[tuple[str, int]]:
    if session is None:
        return []
    return [
        (UtteranceUtil.section_label(session.book, index), index)
        for index, section in enumerate(session.book.sections)
        if section.phrase_groups
    ]


def line_voice_page_count(session: BookSession | None, section: int | None) -> int:
    if session is None or type(section) is not int or not 0 <= section < len(session.book.sections):
        return 0
    count = len(session.book.sections[section].phrase_groups)
    return (count + LINE_VOICE_PAGE_SIZE - 1) // LINE_VOICE_PAGE_SIZE


def line_voice_rows(session: BookSession | None, section: int | None, page: int = 1) -> list[list[int | str]]:
    if (session is None or section is None or type(page) is not int
            or not 1 <= page <= line_voice_page_count(session, section)):
        return []
    start, end = session.book.section_ranges()[section]
    start += (page - 1) * LINE_VOICE_PAGE_SIZE
    rows = []
    for index in range(start, min(start + LINE_VOICE_PAGE_SIZE, end)):
        group = session.book.phrase_groups[index]
        slot = group.voice_index
        voice = session.line_voices.get(slot) if slot > 0 else None
        name = (voice.name or f"Voice {slot}") if voice is not None else "Automatic"
        rows.append([index + 1, name, group.text.strip()])
    return rows
