from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from tts_audiobook_tool import generate_util
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.validation_findings import ValidationFindings
from tts_audiobook_tool.app_types.validation_result import WordErrorResult
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

from generate_files_test_support import StubValidationResult, generate_files_mock_stack, make_generate_state


def test_batch_heading_can_omit_the_leading_divider(monkeypatch) -> None:
    output: list[str] = []
    monkeypatch.setattr(generate_util, "printt", lambda text="": output.append(text))

    GenerateUtil.print_batch_heading([0], show_divider=False)

    assert len(output) == 2
    assert "Processing line 1" in output[0]
    assert output[1] == ""
    assert not any("---" in line for line in output)


@pytest.mark.parametrize("quiet", (True, False))
def test_routine_validation_pass_output_is_optional(monkeypatch, quiet: bool) -> None:
    output: list[str] = []
    monkeypatch.setattr(generate_util, "printt", lambda text="": output.append(text))
    monkeypatch.setattr(generate_util, "print_accent_separator", output.append)
    state = make_generate_state(num_groups=4, max_retries=1)
    sound = Sound(np.zeros(8, dtype=np.float32), 24000)
    passed = WordErrorResult(sound, [], num_words=3, threshold=0)
    failed = WordErrorResult(sound, [], num_words=3, threshold=0, findings=ValidationFindings(["missing word"]))
    warning = WordErrorResult(sound, [], num_words=3, threshold=1, findings=ValidationFindings(possible_truncation=True))
    results = iter([passed, failed, passed, passed, warning])
    transcripts = generate_util.SegmentTranscriptUtil
    options = {"suppress_validation_passes": True} if quiet else {}

    with (
        generate_files_mock_stack(lambda **_: [next(results)]),
        patch.object(transcripts, "from_validation_result", side_effect=lambda **kwargs: f"STT line {kwargs['index'] + 1}"),
        patch.object(transcripts, "make_stt_details_lines", side_effect=lambda info, **_: [info]),
    ):
        save = cast(MagicMock, GenerateUtil.save_sound_and_timing_json)

        def save_segment(_state, _group, index, _result, **_):
            return ("disk full" if index == 3 else "", "saved.wav")

        save.side_effect = save_segment
        with pytest.raises(generate_util.CheckpointWriteError, match="disk full"):
            GenerateUtil.generate_files(state, {0, 1, 2, 3}, batch_size=1, is_regen=False, **options)
        assert [item.kwargs["stt_info"] for item in save.call_args_list] == [
            "STT line 1", "STT line 2", "STT line 2", "STT line 3", "STT line 4",
        ]

    console = "\n".join(output)
    assert ("STT line 1" in console) is not quiet
    assert ("Line 1:" in console) is not quiet
    assert console.count("STT line 2") == 2
    assert "Word error fail" in console and "will retry" in console
    assert "Passed on retry" in console
    assert "Couldn't save file: disk full" in console
    assert "last word may be truncated" in console
    assert "Processing line 1" in console
    assert "Elapsed:" in console and "Lines saved:" in console


def test_clean_quick_generation_retries_then_omits_metrics_summary(monkeypatch) -> None:
    output: list[str] = []
    monkeypatch.setattr(generate_util, "printt", lambda text="": output.append(text))
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        get_tts_model_type=lambda: TtsModelType.require_by_id("none"),
        max_retries=2,
        phrase_groups=[phrase_group],
        sound_segments=sound_segments,
        generate_range_string="all",
        language_code="en",
        save=MagicMock(return_value=""),
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(stt_variant=None, stt_config=None, save_debug_files=False),
        ),
    )
    results = [StubValidationResult(True), StubValidationResult(True), StubValidationResult(False)]
    force_random_seed_values: list[bool] = []

    def generate_and_validate_batch(**kwargs: object) -> list[StubValidationResult]:
        force_random_seed_values.append(kwargs["force_random_seed"])  # type: ignore[arg-type]
        return [results.pop(0)]

    with (
        generate_files_mock_stack(generate_and_validate_batch),
        patch.object(GenerateUtil, "print_batch_heading") as print_batch_heading,
    ):
        did_interrupt = GenerateUtil.generate_files(state, {0}, batch_size=1, is_regen=True)

    assert not did_interrupt
    assert not any("Elapsed:" in text for text in output)
    assert not any("Lines saved:" in text for text in output)
    assert print_batch_heading.call_args_list == [
        # Every batch and retry emits a textual divider. Textual worker logs
        # promote it to a semantic full-width rule; console output keeps it.
        call(indices=[0], voice_index=None, show_divider=True),
        call(indices=[0], voice_index=None, show_divider=True),
        call(indices=[0], voice_index=None, show_divider=True),
    ]
    assert force_random_seed_values == [True, True, True]
    assert results == []
    assert project.generate_range_string == "none"
    project.save.assert_called_once_with()


def test_staged_generation_leaves_project_segments_and_range_alone(monkeypatch) -> None:
    # A staged (reviewable) run saves into the staging directory, prunes takes
    # there instead of in segments/, and does not touch the generation range.
    monkeypatch.setattr(generate_util, "printt", lambda text="": None)
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        get_tts_model_type=lambda: TtsModelType.require_by_id("none"),
        max_retries=1,
        phrase_groups=[phrase_group],
        sound_segments=sound_segments,
        generate_range_string="all",
        language_code="en",
        save=MagicMock(return_value=""),
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(stt_variant=None, stt_config=None, save_debug_files=False),
        ),
    )
    results = [StubValidationResult(True), StubValidationResult(False)]

    with (
        generate_files_mock_stack(lambda **_: [results.pop(0)]),
        patch.object(generate_util, "StagedCheckpoints") as staged,
    ):
        save = cast(MagicMock, GenerateUtil.save_sound_and_timing_json)
        did_interrupt = GenerateUtil.generate_files(
            state, {0}, batch_size=1, is_regen=True, staging_dir="/staging/job"
        )
        save_dirs = [c.kwargs["dir_path"] for c in save.call_args_list]

    assert not did_interrupt
    assert save_dirs == ["/staging/job", "/staging/job"]
    staged.assert_called_once_with("/staging/job")
    assert all(c.kwargs["on_published"] is staged.return_value.record for c in save.call_args_list)
    sound_segments.record_published_segment.assert_not_called()
    assert project.generate_range_string == "all"
    project.save.assert_not_called()


def test_failed_quick_generation_keeps_metrics_summary(monkeypatch) -> None:
    output: list[str] = []
    monkeypatch.setattr(generate_util, "printt", lambda text="": output.append(text))
    phrase_group = PhraseGroup([Phrase("Hello world.", Reason.SENTENCE)])
    sound_segments = MagicMock()
    sound_segments.get_word_error_counts_in_generate_range.return_value = {}
    project = SimpleNamespace(
        get_tts_model_type=lambda: TtsModelType.require_by_id("none"),
        max_retries=0,
        phrase_groups=[phrase_group],
        sound_segments=sound_segments,
        generate_range_string="all",
        language_code="en",
        save=MagicMock(return_value=""),
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(
                stt_variant=None,
                stt_config=None,
                save_debug_files=False,
            ),
        ),
    )

    with generate_files_mock_stack(lambda **_: [StubValidationResult(True)]):
        did_interrupt = GenerateUtil.generate_files(
            state,
            {0},
            batch_size=1,
            is_regen=True,
        )

    assert not did_interrupt
    assert any("Elapsed:" in text for text in output)
    assert any("Lines saved, but with excess word errors:" in text for text in output)
