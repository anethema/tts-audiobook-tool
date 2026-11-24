"""Book text import and line edits for the local browser."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import replace
from itertools import groupby
from pathlib import Path

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.delivery_cue import DeliveryCue
from tts_audiobook_tool.app_types.phrase import PhraseGroup, Reason
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_DEFAULT, MAX_WORDS_PER_SEGMENT_MAX, MAX_WORDS_PER_SEGMENT_MIN
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.attribution import (
    AttributionsFile, RangeAttribution, UtteranceAttribution, UTTERANCE_TEXT_EXCERPT_CHARS,
)
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.text_ops import text_format
from tts_audiobook_tool.text_ops.epub_extractor import EpubExtractor
from tts_audiobook_tool.text_ops.epub_format_reapply import reapply_epub_formatting
from tts_audiobook_tool.text_ops.format_attach import attach_formatting_to_groups
from tts_audiobook_tool.text_ops.heading_end_detector import _text_end_reason, recompute_heading_ends
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.text_ops.text_edit_session import PhraseGroupSplitPoint, TextEditSession
from tts_audiobook_tool.textual.word_substitutions_dialog import normalize_original_key
from tts_audiobook_tool.web_backend import BookSession, WebBackendError, new_book_session


def reapply_browser_formatting(session: BookSession, source: str | None) -> tuple[BookSession, str]:
    path = Path(source or "")
    if path.suffix.lower() != ".epub" or not path.is_file():
        raise WebBackendError("Choose the source EPUB to reapply formatting.")
    book = deepcopy(session.book)
    try:
        result = reapply_epub_formatting(book, str(path))
    except Exception as error:
        raise WebBackendError("Could not read the EPUB formatting.") from error
    if isinstance(result, str):
        raise WebBackendError(result)
    message = (
        f"Applied formatting to {result.num_paired_sections:,} of {result.num_sections:,} sections. "
        f"{result.num_formatted_phrases:,} phrases have formatting."
    )
    if result.num_unmatched_words:
        message += f" {result.num_unmatched_words:,} words differ from the EPUB."
    return replace(session, book=book), message


def apply_export_formatting(checkpoint: Book, current: Book) -> None:
    """Use current formatting only where text and phrase boundaries still match the takes."""
    if len(checkpoint.sections) != len(current.sections):
        return
    pairs = []
    for previous_section, current_section in zip(checkpoint.sections, current.sections, strict=True):
        if len(previous_section.phrase_groups) != len(current_section.phrase_groups):
            return
        for previous, updated in zip(previous_section.phrase_groups, current_section.phrase_groups, strict=True):
            if previous.phrases != updated.phrases:
                return
            pairs.extend(zip(previous.phrases, updated.phrases, strict=True))
    for previous, updated in pairs:
        previous.fmt = updated.fmt


def _segmentation_settings(
    language: str | None, max_words: int, strategy_id: str, dialogue: bool,
) -> BookSegmentationSettings:
    strategy = SegmentationStrategy.from_id(strategy_id)
    if strategy is None:
        raise WebBackendError("Choose a segmentation strategy.")
    if type(max_words) is not int or not MAX_WORDS_PER_SEGMENT_MIN <= max_words <= MAX_WORDS_PER_SEGMENT_MAX:
        raise WebBackendError(f"Enter a maximum word count from {MAX_WORDS_PER_SEGMENT_MIN} to {MAX_WORDS_PER_SEGMENT_MAX}.")
    return BookSegmentationSettings((language or "").strip().lower(), max_words, strategy, dialogue)


def import_book_text(
    source: str, epub_path: str | None, text_path: str | None, pasted_text: str | None,
    language: str | None, max_words: int, strategy_id: str, dialogue: bool,
) -> BookSession:
    """Import an EPUB, UTF-8 text file, or pasted text through the console segmenters."""
    settings = _segmentation_settings(language, max_words, strategy_id, dialogue)
    if source == "epub":
        return _import_epub(epub_path, settings)

    title = ""
    if source == "plain_text":
        path = Path(text_path or "")
        if path.suffix.lower() != ".txt" or not path.is_file():
            raise WebBackendError("Choose a TXT file to import.")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            raise WebBackendError("Could not read the TXT file. Save it as UTF-8 and try again.") from None
        title = path.stem
    elif source == "manual":
        text = pasted_text or ""
    else:
        raise WebBackendError("Choose a text source.")

    groups = _segment_text(text, settings)
    if not groups:
        raise WebBackendError("The source contains no readable text.")
    book = ProjectTextIOUtil.make_import_book(
        groups, settings.strategy, max_words, settings.language_code, dialogue, title, source,
    )
    return new_book_session(book, project_language=settings.language_code)


def _segment_text(text: str, settings: BookSegmentationSettings) -> list[PhraseGroup]:
    language, max_words, strategy, dialogue = settings
    return PhraseGrouper.text_to_groups(
        text, max_words, strategy=strategy, pysbd_lang=language, dialog_segmentation=dialogue,
    )


def _import_epub(path: str | None, settings: BookSegmentationSettings) -> BookSession:
    epub = Path(path or "")
    if epub.suffix.lower() != ".epub" or not epub.is_file():
        raise WebBackendError("Choose an EPUB file to import.")
    language, max_words, strategy, dialogue = settings
    try:
        imported = EpubExtractor.import_epub(str(epub), max_words, strategy, language, dialog_segmentation=dialogue)
    except Exception:
        raise WebBackendError("The EPUB could not be read. Check that the file opens in an EPUB reader.") from None
    if not imported.phrase_groups:
        raise WebBackendError("The EPUB contains no readable text.")
    book = ProjectTextIOUtil.make_import_book(
        imported.phrase_groups, strategy, max_words, language, dialogue,
        imported.book_title.strip() or epub.stem, "epub",
        section_start_indices=imported.section_start_indices,
        section_titles=[chapter.title for chapter in imported.chapters],
    )
    return new_book_session(book, warnings=tuple(imported.significant_warnings), project_language=language)


def section_lines(session: BookSession, section: int) -> tuple[int, list[PhraseGroup]]:
    if type(section) is not int or not 0 <= section < len(session.book.sections):
        raise WebBackendError("Choose a book section.")
    return session.book.section_start_indices()[section], session.book.sections[section].phrase_groups


def can_join_line(session: BookSession, section: int, line: int) -> bool:
    first, groups = section_lines(session, section)
    if not first <= line < first + len(groups) - 1:
        return False
    left, right = groups[line - first:line - first + 2]
    max_words = session.book.segmentation_settings.max_words_per_segment or MAX_WORDS_PER_SEGMENT_DEFAULT
    if (
        left.voice_index != right.voice_index or left.is_dialog != right.is_dialog
        or left.last_reason >= Reason.HEADING_END or left.num_words + right.num_words > max_words
    ):
        return False
    speakers = session.attributions.speaker_of_each_line(session.book)
    return speakers.get(line) == speakers.get(line + 1)


def adjacent_section(session: BookSession, section: int, direction: int) -> int | None:
    sections = session.book.sections
    stop = len(sections) if direction > 0 else -1
    return next((index for index in range(section + direction, stop, direction) if sections[index].phrase_groups), None)


def change_section(
    session: BookSession, section: int, line: int, operation: str,
) -> tuple[BookSession, int]:
    """Change chapter boundaries without moving any book-wide line."""
    first, groups = section_lines(session, section)
    if type(line) is not int or not first <= line < first + len(groups):
        raise WebBackendError("Choose a text line in the current section.")

    sections = deepcopy(session.book.sections)
    if operation == "split":
        if line == first:
            raise WebBackendError("Choose a line after the start of the section.")
        current = sections[section]
        split = line - first
        following = BookSection(phrase_groups=current.phrase_groups[split:])
        current.phrase_groups = current.phrase_groups[:split]
        EpubExtractor.mark_last_phrase_as_section(current.phrase_groups)
        focus = section + 1
        sections.insert(focus, following)
    elif operation in ("join_previous", "join_next"):
        direction = -1 if operation == "join_previous" else 1
        neighbor = adjacent_section(session, section, direction)
        if neighbor is None:
            raise WebBackendError("Choose a section with a neighboring section to join.")
        focus, later = sorted((section, neighbor))
        earlier_section = sections[focus]
        ending = earlier_section.phrase_groups[-1]
        if ending.last_reason == Reason.SECTION_BREAK:
            ending.phrases[-1].reason = _text_end_reason(ending.text)
        earlier_section.phrase_groups.extend(sections.pop(later).phrase_groups)
    else:
        raise WebBackendError("Choose a section edit action.")

    book = replace(session.book, sections=sections)
    recompute_heading_ends(book.sections, book.segmentation_settings.language_code)
    attributions = _preserve_section_attributions(book, session.attributions)
    if attributions != session.attributions:
        directory = session.attribution_dir or session.project_dir
        if SpeakerAssignFileUtil.save_attributions_file(directory, attributions):
            raise WebBackendError("Could not update saved dialogue assignments. The section edit was not applied.")
    return replace(session, book=book, attributions=attributions), focus


def _utterance_fragment(book: Book, original: UtteranceAttribution, lines: tuple[int, ...]) -> UtteranceAttribution:
    if lines == original.lines:
        return original
    text = "".join(book.phrase_groups[line].text for line in lines).strip()
    delivery = original.delivery
    if delivery:
        delivery = delivery.with_boundaries(
            before=lines[0] == original.first_line, after=lines[-1] == original.lines[-1],
        )
    return replace(original, lines=lines, text=text[:UTTERANCE_TEXT_EXCERPT_CHARS], delivery=delivery)


def _preserve_section_attributions(book: Book, previous: AttributionsFile) -> AttributionsFile:
    if not previous.ranges:
        return previous
    plan = AssignPlan.make(book, previous)
    starts = sorted(set(book.section_start_indices()) | {chunk.start for chunk in plan.chunks})
    original_assignments = _attributions_by_line(book, previous)
    ranges = []
    for saved in previous.ranges:
        if saved.is_valid_for(book):
            # A split can divide a spoken utterance between two chapters. Keep
            # the same speaker and confidence on either side of the boundary.
            utterances = tuple(
                _utterance_fragment(book, utterance, tuple(lines))
                for utterance in saved.utterances
                for _, lines in groupby(utterance.lines, key=lambda line: bisect_right(starts, line))
            )
            saved = replace(saved, utterances=utterances)
        ranges.append(saved)
    attributions = AttributionsFile(tuple(ranges))

    # Chapter boundaries can change the native chunk plan. Create only the
    # replacement ranges for which every dialogue line has a known assignment.
    assignments = _attributions_by_line(book, attributions)
    for chunk in AssignPlan.make(book, attributions).remaining:
        dialogue = [line for line in range(chunk.start, chunk.end) if book.phrase_groups[line].is_dialog]
        if any(line not in assignments for line in dialogue):
            continue
        utterances = tuple(
            _utterance_fragment(book, utterance, tuple(lines))
            for utterance, lines in groupby(dialogue, key=assignments.__getitem__)
        )
        attributions = attributions.with_range(RangeAttribution(
            chunk.start, chunk.end, UtteranceUtil.range_text_hash(book, chunk.start, chunk.end), utterances,
        ))

    # Compare projected line cues because a split changes utterance boundaries.
    before = {line: _attribution_for_line(item, line) for line, item in original_assignments.items()}
    after = {line: _attribution_for_line(item, line) for line, item in _attributions_by_line(book, attributions).items()}
    return attributions if before == after and attributions != previous else previous


def _attribution_for_line(item: UtteranceAttribution, line: int) -> tuple[str, bool, DeliveryCue | None]:
    cue = item.delivery
    if cue:
        cue = cue.with_boundaries(before=line == item.first_line, after=line == item.lines[-1])
    return item.speaker, item.uncertain, cue


def _attributions_by_line(book: Book, attributions: AttributionsFile) -> dict[int, UtteranceAttribution]:
    return {line: utterance for utterance in attributions.by_first_line(book).values() for line in utterance.lines}


def change_line(
    session: BookSession, section: int, line: int, operation: str, *, text: str = "", boundary: int | None = None,
) -> BookSession:
    """Apply one console-compatible edit and keep only still-valid dialogue assignments."""
    first, groups = section_lines(session, section)
    if type(line) is not int or not first <= line < first + len(groups):
        raise WebBackendError("Choose a text line in the current section.")
    staged = TextEditSession(session.book)
    item = staged.sections[section].phrase_groups[line - first]
    if operation == "edit":
        if not text.strip():
            raise WebBackendError("Enter text for this line or use Delete line.")
        settings = session.book.segmentation_settings
        result = staged.update_phrase_group_text(
            item.item_id, text, settings.max_words_per_segment or MAX_WORDS_PER_SEGMENT_DEFAULT, settings.language_code,
        )
    elif operation == "split":
        if type(boundary) is not int or not 0 < boundary < len(item.phrase_group.phrases):
            raise WebBackendError("Choose a phrase boundary to split this line.")
        result = staged.split_phrase_group(item.item_id, PhraseGroupSplitPoint(boundary))
    elif operation == "delete":
        if len(session.book.phrase_groups) <= 1:
            raise WebBackendError("The book needs at least one line of text.")
        result = staged.delete_phrase_groups({item.item_id})
    elif operation == "join":
        if not can_join_line(session, section, line):
            raise WebBackendError("The next line must share the voice and dialogue context and fit within the word limit.")
        following = staged.sections[section].phrase_groups[line - first + 1]
        item.phrase_group.phrases = [*item.phrase_group.phrases, *following.phrase_group.phrases]
        result = staged.delete_phrase_groups({following.item_id})
    else:
        raise WebBackendError("Choose a text edit action.")
    if not result.changed:
        return session

    return _changed_book(session, staged.to_book())


def set_titles(session: BookSession, section: int, book_title: str, section_title: str) -> BookSession:
    section_lines(session, section)
    title = book_title.strip()
    subtitle = section_title.strip()
    if title == session.book.title and subtitle == session.book.sections[section].title:
        return session
    if subtitle == session.book.sections[section].title:
        return replace(session, book=replace(session.book, title=title))

    book = deepcopy(session.book)
    book.title = title
    book.sections[section].title = subtitle
    return _changed_book(session, book)


def set_project_language(session: BookSession, language: str) -> BookSession:
    language = language.strip().lower()
    return session if language == session.language_code else replace(session, project_language=language)


def set_word_substitutions(session: BookSession, rows: list[list[str | None]]) -> BookSession:
    substitutions: dict[str, str] = {}
    keys: set[str] = set()
    for number, row in enumerate(rows, 1):
        if len(row) != 2 or any(value is not None and not isinstance(value, str) for value in row):
            raise WebBackendError("Each pronunciation row needs an original word and a replacement.")
        original, replacement = ((value or "").strip() for value in row)
        if not original and not replacement:
            continue
        if not original or not replacement:
            raise WebBackendError(f"Enter both words in pronunciation row {number}.")
        word, error = normalize_original_key(original)
        if error:
            raise WebBackendError(f"Pronunciation row {number}. {error}.")
        key = word.casefold()
        if key in keys:
            raise WebBackendError(f"The pronunciation list contains more than one entry for {word!r}.")
        keys.add(key)
        substitutions[word] = replacement
    if list(substitutions.items()) == list(session.word_substitutions.items()):
        return session
    return replace(session, word_substitutions=substitutions)


def resegment_book(
    session: BookSession, language: str | None, max_words: int, strategy_id: str, dialogue: bool,
) -> BookSession:
    """Rebuild lines in the current text, preserving sections and voice boundaries."""
    settings = _segmentation_settings(language, max_words, strategy_id, dialogue)
    sections = [_resegment_section(section, settings) for section in session.book.sections]
    book = replace(session.book, sections=sections, segmentation_settings=settings)
    updated = _changed_book(session, book)
    if updated is not session and updated.language_code != session.language_code:
        updated = replace(updated, project_language=session.language_code)
    return updated


_PRESERVED_BREAKS = {Reason.SPACE_BREAK, Reason.SECTION_BREAK}


def _resegment_section(section: BookSection, settings: BookSegmentationSettings) -> BookSection:
    groups: list[PhraseGroup] = []
    run: list[PhraseGroup] = []
    for group in section.phrase_groups:
        if run and group.voice_index != run[-1].voice_index:
            groups.extend(_resegment_run(run, settings))
            run = []
        run.append(group)
        if group.last_reason in _PRESERVED_BREAKS:
            groups.extend(_resegment_run(run, settings))
            run = []
    if run:
        groups.extend(_resegment_run(run, settings))
    return replace(section, phrase_groups=groups)


def _resegment_run(groups: list[PhraseGroup], settings: BookSegmentationSettings) -> list[PhraseGroup]:
    phrases = PhraseGroup.flatten_groups(groups)
    text = "".join(phrase.text for phrase in phrases)
    if not text.strip():
        return deepcopy(groups)
    new_groups = _segment_text(text, settings)
    if not new_groups or "".join(group.text for group in new_groups).split() != text.split():
        raise WebBackendError("The text could not be resegmented without changing its words.")
    if any(phrase.fmt is not None for phrase in phrases):
        styled = text_format.StyledText.join([
            text_format.StyledText(phrase.text, text_format.masks_for(phrase.fmt, phrase.text) or [])
            for phrase in phrases
        ], "")
        attach_formatting_to_groups(new_groups, styled)
    for group in new_groups:
        group.voice_index = groups[0].voice_index
    if groups[-1].last_reason in _PRESERVED_BREAKS:
        new_groups[-1].phrases[-1].reason = groups[-1].last_reason
    return new_groups


def _changed_book(session: BookSession, book: Book) -> BookSession:
    recompute_heading_ends(book.sections, book.segmentation_settings.language_code)
    if TextEditSession.make_snapshot(book) == TextEditSession.make_snapshot(session.book):
        return session

    original = session.book.phrase_groups
    updated = book.phrase_groups
    valid = tuple(
        assignment for assignment in session.attributions.valid_ranges(book)
        if assignment.end <= len(original) and all(
            (old.text, old.is_dialog) == (new.text, new.is_dialog)
            for old, new in zip(original[assignment.start:assignment.end], updated[assignment.start:assignment.end])
        )
    )
    removed = len(session.attributions.ranges) - len(valid)
    if not removed:
        return replace(session, book=book)
    attributions = replace(session.attributions, ranges=valid)
    directory = session.attribution_dir or session.project_dir
    if SpeakerAssignFileUtil.save_attributions_file(directory, attributions):
        raise WebBackendError("Could not update saved dialogue assignments. The text edit was not applied.")
    notice = "Some dialogue assignments need to be refreshed after the text edit."
    return replace(session, book=book, attributions=attributions, attribution_notice=notice)
