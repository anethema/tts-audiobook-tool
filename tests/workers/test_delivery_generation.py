from types import SimpleNamespace
from typing import cast

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types.delivery_cue import DeliveryCue
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.validation_result import WordErrorResult
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.transcriber import Transcriber
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.breeze_base_model import BreezeBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.validator import Validator


def test_delivery_generation_keeps_cues_with_indices_and_validates_original_text(monkeypatch):
    model_id = "breeze_tts_2_local"
    model_type = TtsModelType.require_by_id(model_id)
    project = Project(tts_model_type=model_id)
    prefs = Prefs()
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))
    groups = [
        PhraseGroup([Phrase("Stay there!", Reason.SENTENCE)], delivery=DeliveryCue(direction="Urgently")),
        PhraseGroup([Phrase("It moved.", Reason.SENTENCE)]),
        PhraseGroup([Phrase("Did you see it?", Reason.SENTENCE)], delivery=DeliveryCue(before="gasps")),
    ]
    backend_calls = []
    validated = []

    def generate(*, prompts, force_random_seed, delivery_directions=None, **kwargs):
        backend_calls.append((prompts, force_random_seed, delivery_directions))
        return [Sound(np.full(4, len(prompt), dtype=np.float32), 24000) for prompt in prompts]

    def validate(sound, source, words, language_code, **kwargs):
        validated.append((source, sound.data[0]))
        return WordErrorResult(sound, words, num_words=3, threshold=0)

    monkeypatch.setattr(Tts, "_type", model_type)
    monkeypatch.setattr(Tts, "get_instance", lambda: SimpleNamespace(
        prepare_text_for_inference=BreezeBaseModel.prepare_text_for_inference,
        generate_using_project=generate,
    ))
    monkeypatch.setattr(Stt, "should_skip", lambda *args: "")
    monkeypatch.setattr(Transcriber, "transcribe_to_words", lambda *args: [])
    monkeypatch.setattr(SoundPipeline, "apply_generate_post_processing_with_info",
                        lambda sound: (sound, None, None, sound.duration))
    monkeypatch.setattr(Validator, "validate", validate)

    for indices, retry in [([2, 0, 1], False), ([2], True)]:
        results = GenerateUtil.generate_and_validate_batch(
            state, indices, groups, prefs.stt_variant, prefs.stt_config,
            force_random_seed=retry, is_realtime=retry, voice_selection_index=0,
        )
        assert all(isinstance(result, WordErrorResult) for result in results)

    tagged = "(gasps) Did you see it?"
    assert backend_calls == [
        ([tagged, "Stay there!", "It moved."], False, ["", "Urgently", ""]),
        ([tagged], True, None),
    ]
    assert validated == [
        ("Did you see it?", len(tagged)), ("Stay there!", len("Stay there!")),
        ("It moved.", len("It moved.")), ("Did you see it?", len(tagged)),
    ]
    assert [group.text for group in groups] == ["Stay there!", "It moved.", "Did you see it?"]
    assert SegmentTranscriptUtil.make_inference_prompt(project, groups[2].text, groups[2].delivery) == tagged
