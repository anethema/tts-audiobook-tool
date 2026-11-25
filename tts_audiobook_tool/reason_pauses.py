from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from tts_audiobook_tool.app_types.phrase import Phrase, Reason


# Pacing cues at the join only. Punctuation in the middle of a take is performed by the voice.
_DELIBERATE_END = re.compile(
    r"""(?:…|\.{3}|\b(?:he|she|they|we|i) (?:pauses?|paused|hesitates?|hesitated)\.)["'”’»)]*\s*$""",
    re.I,
)
_CONTINUATION_START = re.compile(r"(?:and|but|or|so)\s+", re.I)


@dataclass(frozen=True)
class ReasonPauses:
    """A named pause-duration policy for phrase segmentation reasons."""

    id: str
    label: str
    menu_label: str
    description: str
    pauses: Mapping[Reason, float]

    def get_pause_for(self, reason: Reason) -> float:
        """Return the pause duration in seconds for ``reason``."""
        return self.pauses[reason]

    def get_join_pause(self, phrase: Phrase, following_text: str) -> float:
        """Target gap after a completed take, before its audible edge silence is deducted."""
        if self.id == "natural" and phrase.reason is Reason.SENTENCE and following_text:
            if _DELIBERATE_END.search(phrase.text):
                return 1.0
            if phrase.text.rstrip().endswith(".") and _CONTINUATION_START.match(following_text.lstrip()):
                return 0.55
        return self.get_pause_for(phrase.reason)

    def _clone(self) -> "ReasonPauses":
        # The `pauses` mapping is a MappingProxyType, which copy.copy/deepcopy
        # cannot handle (TypeError: cannot pickle 'mappingproxy' object).
        # Rebuild the dataclass with a plain dict copy instead.
        return ReasonPauses(
            id=self.id,
            label=self.label,
            menu_label=self.menu_label,
            description=self.description,
            pauses=dict(self.pauses),
        )

    def __copy__(self) -> "ReasonPauses":
        return self._clone()

    def __deepcopy__(self, memo) -> "ReasonPauses":
        return self._clone()


NORMAL_REASON_PAUSES = MappingProxyType(
    {
        Reason.UNDEFINED: 1.0,
        Reason.WORD: 0.1,
        Reason.PHRASE_QUOTE_END: 0.3,
        Reason.PHRASE: 0.5,
        Reason.SENTENCE: 0.9,
        Reason.PARAGRAPH: 1.2,
        Reason.HEADING_END: 2.0,
        Reason.SPACE_BREAK: 2.5,
        Reason.SECTION_BREAK: 3.0,
    }
)

NATURAL_REASON_PAUSES = MappingProxyType({
    **NORMAL_REASON_PAUSES,
    Reason.SENTENCE: 0.7,
    Reason.PARAGRAPH: 0.9,
})

SHORTER_REASON_PAUSES = MappingProxyType(
    {
        Reason.UNDEFINED: 1.0,
        Reason.WORD: 0.1,
        Reason.PHRASE_QUOTE_END: 0.2,
        Reason.PHRASE: 0.3,
        Reason.SENTENCE: 0.6,
        Reason.PARAGRAPH: 0.9,
        Reason.HEADING_END: 1.5,
        Reason.SPACE_BREAK: 2.0,
        Reason.SECTION_BREAK: 2.5,
    }
)

def make_pauses_description(pauses: Mapping[Reason, float]) -> str:
    """User-facing summary of the most common pause durations in ``pauses``."""
    return (
        f"phrase {pauses[Reason.PHRASE]:.1f}s, "
        f"sentence {pauses[Reason.SENTENCE]:.1f}s, "
        f"paragraph {pauses[Reason.PARAGRAPH]:.1f}s, etc."
    )

class ReasonPauseTypes(Enum):
    """Enumerates the available pause-duration policies."""

    NATURAL = ReasonPauses(
        id="natural",
        label="Natural",
        menu_label="natural",
        description="sentence 0.7s (0.55s for continuations, 1.0s for hesitation), paragraph 0.9s",
        pauses=NATURAL_REASON_PAUSES,
    )
    NORMAL = ReasonPauses(
        id="normal",
        label="Normal",
        menu_label="normal",
        description=make_pauses_description(NORMAL_REASON_PAUSES),
        pauses=NORMAL_REASON_PAUSES,
    )
    SHORTER = ReasonPauses(
        id="shorter",
        label="Shorter",
        menu_label="shorter",
        description=make_pauses_description(SHORTER_REASON_PAUSES),
        pauses=SHORTER_REASON_PAUSES,
    )

    @staticmethod
    def default() -> ReasonPauseTypes:
        return ReasonPauseTypes.NATURAL

    @staticmethod
    def get_by_id(id: str) -> ReasonPauseTypes | None:
        for item in ReasonPauseTypes:
            if item.value.id == id:
                return item
        return None
