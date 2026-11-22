from pathlib import Path

from ebooklib import epub
import pytest

from tts_audiobook_tool.app_types import Book, BookSection, BookSegmentationSettings, SegmentationStrategy
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution, UtteranceAttribution
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.text_ops.prompt_normalizer import PromptNormalizer
from tts_audiobook_tool.web_backend import WebBackendError, close_book_session, new_book_session
from tts_audiobook_tool.web_render import RenderVoice
from tts_audiobook_tool.web_text import (
    apply_export_formatting, can_join_line, change_line, change_section, import_book_text, reapply_browser_formatting,
    resegment_book, set_project_language, set_titles, set_word_substitutions,
)


@pytest.mark.parametrize("source", ["plain_text", "manual"])
def test_imports_plain_text_with_configured_segmentation(tmp_path, source):
    text = "Rain fell. “Come in,” said Rue.\n\n" + "The rain kept falling " * 10 + "."
    path = tmp_path / "A rainy day.txt"
    path.write_text(text, encoding="utf-8-sig")
    session = import_book_text(source, None, str(path), text, "  EN-GB ", 20, "sentence", True)
    try:
        book = session.book
        assert book.text_source_kind == source
        assert book.title == (path.stem if source == "plain_text" else "")
        assert book.segmentation_settings == ("en-gb", 20, SegmentationStrategy.SENTENCE, True)
        assert [book.phrase_groups[index].text.strip() for index in book.dialog_line_indices()] == ["“Come in,”"]
        assert max(group.num_words for group in book.phrase_groups) <= 20
    finally:
        close_book_session(session)


def test_epub_text_import_keeps_sections_and_respects_dialog_toggle(tmp_path):
    path = tmp_path / "book.epub"
    book = epub.EpubBook()
    book.set_identifier("sample-book")
    book.set_title("A rainy day")
    book.set_language("en")
    chapters = [epub.EpubHtml(title=f"Chapter {n}", file_name=f"chapter{n}.xhtml", lang="en") for n in (1, 2)]
    for chapter in chapters:
        chapter.content = "<h1>Chapter</h1><p>“<em>Come in,</em>” said Rue.</p>"
        book.add_item(chapter)
    book.spine = chapters
    book.toc = [epub.Link(chapter.file_name, chapter.title, chapter.file_name) for chapter in chapters]
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)

    session = import_book_text("epub", str(path), None, None, "", 27, "multi", False)
    try:
        assert session.book.title == "A rainy day"
        assert [section.title for section in session.book.sections] == ["Chapter 1", "Chapter 2"]
        assert session.book.segmentation_settings == ("", 27, SegmentationStrategy.MULTI_SENTENCE, False)
        assert session.book.dialog_line_indices() == []
        original_text = [group.text for group in session.book.phrase_groups]
        for group in session.book.phrase_groups:
            for phrase in group.phrases:
                phrase.fmt = None
        updated, _ = reapply_browser_formatting(session, str(path))
        assert [group.text for group in updated.book.phrase_groups] == original_text
        assert not any(phrase.fmt for group in session.book.phrase_groups for phrase in group.phrases)
        assert any(phrase.fmt for group in updated.book.phrase_groups for phrase in group.phrases)
        apply_export_formatting(session.book, updated.book)
        assert any(phrase.fmt for group in session.book.phrase_groups for phrase in group.phrases)
    finally:
        close_book_session(session)


def test_spot_edit_preserves_book_metadata_and_only_valid_attributions():
    book = Book([
        BookSection([PhraseGroup([Phrase("“Come in,”\n", Reason.SECTION_BREAK)], voice_index=2, is_dialog=True)], "One"),
        BookSection([PhraseGroup([Phrase("“I will.”", Reason.SECTION_BREAK)], is_dialog=True)], "Two"),
    ], title="A rainy day", text_source_kind="epub")
    attributions = AttributionsFile(tuple(
        RangeAttribution(index, index + 1, UtteranceUtil.range_text_hash(book, index, index + 1), (
            UtteranceAttribution((index,), "RUE"),
        )) for index in range(2)
    ))
    roster = Roster((RosterCharacter("RUE", "Rue"),))
    session = new_book_session(book, roster=roster, attributions=attributions)
    try:
        edited = change_line(session, 0, 0, "edit", text="“Knock first.”\n\n")
        assert session.book.phrase_groups[0].text == "“Come in,”\n"
        group = edited.book.phrase_groups[0]
        assert (group.text, group.last_reason, group.voice_index, group.is_dialog) == (
            "“Knock first.”\n", Reason.SECTION_BREAK, 2, True,
        )
        assert edited.book.title == book.title
        assert edited.book.text_source_kind == "epub"
        assert edited.roster == roster
        assert edited.attributions.ranges == attributions.ranges[1:]
        assert SpeakerAssignFileUtil.load_attributions_file(edited.attribution_dir) == edited.attributions
        assert Path(edited.project_dir).is_dir()
    finally:
        close_book_session(session)


def test_split_and_delete_use_original_phrase_boundaries():
    book = Book([BookSection([PhraseGroup([
        Phrase("“First. ", Reason.SENTENCE), Phrase("Second.”", Reason.SECTION_BREAK),
    ], voice_index=3, is_dialog=True)])])
    session = new_book_session(book)
    try:
        split = change_line(session, 0, 0, "split", boundary=1)
        assert [group.text for group in split.book.phrase_groups] == ["“First. ", "Second.”"]
        assert all(group.voice_index == 3 and group.is_dialog for group in split.book.phrase_groups)
        deleted = change_line(split, 0, 0, "delete")
        assert [group.text for group in deleted.book.phrase_groups] == ["Second.”"]
        with pytest.raises(WebBackendError, match="at least one line"):
            change_line(deleted, 0, 0, "delete")
    finally:
        close_book_session(session)


def test_join_preserves_phrases_and_requires_a_matching_speaker():
    section = BookSection([
        PhraseGroup([Phrase("“First.” ", Reason.SENTENCE, "<i>“First.”</i> ")], voice_index=2, is_dialog=True),
        PhraseGroup([Phrase("“Second.”", Reason.SECTION_BREAK)], voice_index=2, is_dialog=True),
    ])
    book = Book([section])
    assignments = AttributionsFile((RangeAttribution(0, 2, UtteranceUtil.range_text_hash(book, 0, 2), (
        UtteranceAttribution((0,), "RUE"), UtteranceAttribution((1,), "MILO"),
    )),))
    session = new_book_session(book, attributions=assignments)
    try:
        assert not can_join_line(session, 0, 0)
        with pytest.raises(WebBackendError, match="dialogue context"):
            change_line(session, 0, 0, "join")
    finally:
        close_book_session(session)

    session = new_book_session(book)
    try:
        assert can_join_line(session, 0, 0)
        joined = change_line(session, 0, 0, "join")
        group, = joined.book.phrase_groups
        assert group.phrases == [original.phrases[0] for original in section.phrase_groups]
        assert group.phrases[0].fmt == "<i>“First.”</i> "
        assert group.is_dialog and group.voice_index == 2
        assert len(session.book.phrase_groups) == 2
    finally:
        close_book_session(session)


def test_chapter_edits_preserve_line_assignments_and_formatting():
    book = Book([
        BookSection([
            PhraseGroup([Phrase("“First.\n", Reason.PARAGRAPH, "<i>“First.</i>\n")], voice_index=4, is_dialog=True),
            PhraseGroup([Phrase("Second.”", Reason.SECTION_BREAK)], voice_index=2, is_dialog=True),
        ], "Original chapter"),
        BookSection([PhraseGroup([Phrase("Last.", Reason.SECTION_BREAK)])], "Last chapter"),
    ], title="A book", text_source_kind="epub")
    assignment = RangeAttribution(0, 3, UtteranceUtil.range_text_hash(book, 0, 3), (
        UtteranceAttribution((0, 1), "RUE", uncertain=True),
    ))
    session = new_book_session(book, attributions=AttributionsFile((assignment,)), line_voices={4: RenderVoice(name="Guest")})
    try:
        with pytest.raises(WebBackendError, match="after the start"):
            change_section(session, 0, 0, "split")
        with pytest.raises(WebBackendError, match="neighboring section"):
            change_section(session, 0, 0, "join_previous")

        split, focus = change_section(session, 0, 1, "split")
        assert focus == 1
        assert split.book.section_start_indices() == [0, 1, 2]
        assert [chapter.title for chapter in split.book.sections] == ["Original chapter", "", "Last chapter"]
        assert [item.text for item in split.book.phrase_groups] == [item.text for item in book.phrase_groups]
        assert [item.voice_index for item in split.book.phrase_groups] == [4, 2, 0]
        assert split.line_voices == session.line_voices
        assert split.book.phrase_groups[0].phrases[0].fmt == "<i>“First.</i>\n"
        assert split.book.phrase_groups[0].last_reason is Reason.SECTION_BREAK
        assert session.book.phrase_groups[0].last_reason is Reason.PARAGRAPH
        assert {line: item.lines for line, item in split.attributions.by_first_line(split.book).items()} == {0: (0,), 1: (1,)}
        assert all(item.uncertain for item in split.attributions.by_first_line(split.book).values())
        assert SpeakerAssignFileUtil.load_attributions_file(session.attribution_dir) == split.attributions

        joined, focus = change_section(split, 1, 1, "join_previous")
        assert focus == 0
        assert [section.title for section in joined.book.sections] == ["Original chapter", "Last chapter"]
        assert joined.book.phrase_groups[0].last_reason is Reason.PARAGRAPH
        assert joined.attributions.speaker_of_each_line(joined.book) == {0: "RUE", 1: "RUE"}
        final, focus = change_section(joined, 0, 1, "join_next")
        assert focus == 0
        assert [section.title for section in final.book.sections] == ["Original chapter"]
        assert [item.voice_index for item in final.book.phrase_groups] == [4, 2, 0]
    finally:
        close_book_session(session)


def test_chapter_split_keeps_completed_dialogue_plan_available():
    groups = [
        PhraseGroup([Phrase("“" + "word " * 1100 + "”\n", Reason.SENTENCE)], is_dialog=True)
        for _ in range(6)
    ]
    book = Book([BookSection(groups, "First")])
    original_plan = AssignPlan.make(book, AttributionsFile())
    attributions = AttributionsFile(tuple(
        RangeAttribution(
            chunk.start, chunk.end, UtteranceUtil.range_text_hash(book, chunk.start, chunk.end),
            tuple(UtteranceAttribution((line,), "RUE" if line % 2 else "MILO") for line in range(chunk.start, chunk.end)),
        )
        for chunk in original_plan.chunks
    ))
    session = new_book_session(book, attributions=attributions)
    try:
        updated, _ = change_section(session, 0, 4, "split")
        new_plan = AssignPlan.make(updated.book, updated.attributions)
        assert [(chunk.start, chunk.end) for chunk in original_plan.chunks] == [(0, 3), (3, 6)]
        assert [(chunk.start, chunk.end) for chunk in new_plan.chunks] == [(0, 4), (4, 6)]
        assert new_plan.is_complete
        assert updated.attributions.speaker_of_each_line(updated.book) == attributions.speaker_of_each_line(book)
        assert SpeakerAssignFileUtil.load_attributions_file(session.attribution_dir) == updated.attributions
    finally:
        close_book_session(session)


def test_chapter_split_never_restores_overridden_historical_dialogue():
    book = Book([BookSection([
        PhraseGroup([Phrase("“First. ", Reason.SENTENCE)], is_dialog=True),
        PhraseGroup([Phrase("Second.” ", Reason.SENTENCE)], is_dialog=True),
        PhraseGroup([Phrase("End.", Reason.SECTION_BREAK)]),
    ])])
    assignments = AttributionsFile((
        RangeAttribution(0, 3, UtteranceUtil.range_text_hash(book, 0, 3), (UtteranceAttribution((0, 1), "RUE"),)),
        RangeAttribution(0, 2, UtteranceUtil.range_text_hash(book, 0, 2), (UtteranceAttribution((0,), "MILO"),)),
    ))
    session = new_book_session(book, attributions=assignments)
    try:
        updated, _ = change_section(session, 0, 1, "split")
        assert updated.attributions.speaker_of_each_line(updated.book) == {0: "MILO"}
    finally:
        close_book_session(session)


def test_resegment_preserves_formatting_scene_breaks_and_manual_voice_runs():
    book = Book([
        BookSection([
            PhraseGroup([Phrase("Rain fell. Wind picked up. ", Reason.SPACE_BREAK, "<i>Rain fell.</i> Wind picked up. ")]),
            PhraseGroup([Phrase("Rue closed the door. ", Reason.SENTENCE)], voice_index=3),
            PhraseGroup([Phrase("Milo walked home.", Reason.SECTION_BREAK)]),
        ], "One"),
    ], title="A rainy day", text_source_kind="epub",
        segmentation_settings=BookSegmentationSettings("fr", 40, SegmentationStrategy.SENTENCE_PLUS, False))
    session = new_book_session(book)
    try:
        renamed = set_titles(session, 0, " The rain stops ", " The door ")
        cleared = set_project_language(renamed, " ")
        updated = resegment_book(cleared, "en", 20, "sentence", False)
        groups = updated.book.phrase_groups
        assert [group.text for group in groups] == [
            "Rain fell. ", "Wind picked up. ", "Rue closed the door. ", "Milo walked home.",
        ]
        assert [group.voice_index for group in groups] == [0, 0, 3, 0]
        assert groups[0].phrases[0].fmt == "<i>Rain fell.</i> "
        assert [groups[1].last_reason, groups[-1].last_reason] == [Reason.SPACE_BREAK, Reason.SECTION_BREAK]
        assert updated.book.title == "The rain stops"
        assert updated.book.sections[0].title == "The door"
        assert updated.book.segmentation_settings == ("en", 20, SegmentationStrategy.SENTENCE, False)
        assert updated.language_code == ""
        assert session.book.title == "A rainy day"
        assert session.book.segmentation_settings.language_code == "fr"
    finally:
        close_book_session(session)


def test_resegment_discards_assignment_if_dialogue_detection_changes():
    source = "“Open up.”"
    original = import_book_text("manual", None, None, source, "en", 40, "sentence", True)
    assert original.book.dialog_line_indices() == [0]
    assignment = RangeAttribution(0, 1, UtteranceUtil.range_text_hash(original.book, 0, 1), (
        UtteranceAttribution((0,), "RUE"),
    ))
    session = new_book_session(original.book, attributions=AttributionsFile((assignment,)))
    close_book_session(original)
    try:
        updated = resegment_book(session, "en", 40, "sentence", False)
        assert updated.book.phrase_groups[0].text == source
        assert updated.book.dialog_line_indices() == []
        assert updated.attributions.ranges == ()
        assert SpeakerAssignFileUtil.load_attributions_file(updated.attribution_dir) == updated.attributions
    finally:
        close_book_session(session)


def test_word_substitutions_share_console_key_rules_and_keep_row_order():
    session = import_book_text("manual", None, None, "Malaz and Darujhistan.", "en", 40, "sentence", False)
    try:
        updated = set_word_substitutions(session, [
            ["  (Malaz) ", "Ma-laz"], ["Darujhistan", "Da-roo-jis-tan"], [None, None],
        ])
        assert list(updated.word_substitutions.items()) == [("Malaz", "Ma-laz"), ("Darujhistan", "Da-roo-jis-tan")]
        spoken = PromptNormalizer.apply_prompt_word_substitutions(
            session.book.phrase_groups[0].text, updated.word_substitutions, updated.language_code,
        )
        assert spoken == "Ma-laz and Da-roo-jis-tan."
        assert updated.book is session.book
        with pytest.raises(WebBackendError, match="more than one entry"):
            set_word_substitutions(updated, [["Malaz", "First"], ["malaz", "Second"]])
        with pytest.raises(WebBackendError, match="contain spaces"):
            set_word_substitutions(updated, [["two words", "Single"]])
        assert set_word_substitutions(updated, [[None, None]]).word_substitutions == {}
    finally:
        close_book_session(session)
