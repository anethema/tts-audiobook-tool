"""
`Roster`: a book's speaking characters, as determined by the full-book LLM pass.
The roster is the canonical speaker list: dialog attributions use its character
ids (or one of the reserved answers in `speaker_assign_types`).

On disk (the keys the prototype's cache already uses):

    {
        "narration": {"person": "first", "notes": "...", "voice_hint": "..."},
        "characters": [
            {
                "id": "BARON",                  # canonical speaker label
                "name": "Baron de Verdira",
                "aliases": ["the baron", "pèren"],
                "gender": "male",
                "description": "...",
                "voice_hint": "elderly; deep, gravelly; slow, imperious",
                "voice_direction": "Slightly deeper tone.",
                "is_narrator": false,
                "prominence": "major",
                "merged_ids": ["OLD_BARON_ID"]    # optional user-confirmed redirects
            }
        ]
    }

Names, IDs, casting hints, voice directions and narration notes are
spoiler-free. LLM-facing character aliases and descriptions are complete and
may contain spoilers. `voice_hint` (what kind of reference
voice suits the character or narration) is shown only to the user, not sent
with the dialog attribution requests. `voice_direction` is a stable performance
baseline for a character. Instances are immutable; edits make a new instance with
`dataclasses.replace`.

`RosterFile` is the full-book pass's saved result: the roster plus provenance,
stored as `[project]/speaker_assign/roster.json` (see SpeakerAssignFileUtil).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from tts_audiobook_tool.speaker_assign.speaker_assign_types import RESERVED_SPEAKER_IDS


# Uppercase letters and digits in underscore-separated words: no spaces, no
# leading, trailing or repeated underscores.
ROSTER_ID_PATTERN = re.compile(r"[A-Z0-9]+(?:_[A-Z0-9]+)*")
NEUTRAL_CHARACTER_VOICE_DIRECTION = "Neutral delivery."
CHARACTER_VOICE_DIRECTIONS = (
    NEUTRAL_CHARACTER_VOICE_DIRECTION,
    "Slightly warmer tone.",
    "Slightly softer tone.",
    "Slightly brighter tone.",
    "Slightly deeper tone.",
    "Slightly more measured pace.",
)
MAX_CHARACTER_VOICE_DIRECTION_CHARS = 120
_VOICE_DIRECTION_PUNCTUATION = " ,.'’–-"


def is_current_character_voice_direction(value: object) -> bool:
    """Check whether a roster direction belongs to the current subtle casting palette."""
    return isinstance(value, str) and value in CHARACTER_VOICE_DIRECTIONS


def normalize_voice_direction(value: object) -> str:
    """Clean stored directions, including short plain-language legacy values."""
    direction = " ".join(value.split()) if isinstance(value, str) else ""
    if (
        len(direction) <= MAX_CHARACTER_VOICE_DIRECTION_CHARS
        and any(char.isalpha() for char in direction)
        and all(char.isalpha() or char in _VOICE_DIRECTION_PUNCTUATION for char in direction)
    ):
        return direction
    return ""


def name_key(text: str) -> str:
    """A character name or alias normalized for matching (whitespace collapsed, casefolded)."""
    return " ".join(text.split()).casefold()


class Prominence(Enum):
    """How much dialog a character has (the LLM's overall impression)."""
    MAJOR = "major"
    SUPPORTING = "supporting"
    MINOR = "minor"
    UNKNOWN = ""

    @property
    def rank(self) -> int:
        """Higher means more dialog; UNKNOWN is lowest."""
        return _PROMINENCE_RANKS[self]

    @staticmethod
    def from_value(value: Any) -> Prominence:
        """Tolerant parse; anything unrecognized is UNKNOWN."""
        if isinstance(value, str):
            for member in Prominence:
                if member.value == value.strip().lower():
                    return member
        return Prominence.UNKNOWN


_PROMINENCE_RANKS = {
    Prominence.MAJOR: 3,
    Prominence.SUPPORTING: 2,
    Prominence.MINOR: 1,
    Prominence.UNKNOWN: 0,
}


class NarrativePerson(Enum):
    FIRST = "first"
    THIRD = "third"
    MIXED = "mixed"
    UNKNOWN = ""

    @staticmethod
    def from_value(value: Any) -> NarrativePerson:
        """Tolerant parse; anything unrecognized is UNKNOWN."""
        if isinstance(value, str):
            for member in NarrativePerson:
                if member.value == value.strip().lower():
                    return member
        return NarrativePerson.UNKNOWN


@dataclass(frozen=True)
class RosterCharacter:
    """
    A speaking character. `id` must match ROSTER_ID_PATTERN and not be a
    reserved answer (ValueError otherwise). `gender` is the LLM's free text
    (eg "male", "female", "unknown").
    """
    id: str
    name: str
    aliases: tuple[str, ...] = ()
    gender: str = "unknown"
    description: str = ""
    is_narrator: bool = False
    prominence: Prominence = Prominence.UNKNOWN
    voice_hint: str = ""
    merged_ids: tuple[str, ...] = ()
    voice_direction: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not ROSTER_ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"Invalid roster id {self.id!r}: expected eg BARON or OLD_DOC")
        if self.id in RESERVED_SPEAKER_IDS:
            raise ValueError(f"Roster id {self.id!r} is reserved")
        if not isinstance(self.name, str):
            raise ValueError(f"Roster character {self.id}: name must be a string")
        # Own an immutable copy (callers may pass a list).
        object.__setattr__(self, "aliases", tuple(self.aliases))
        if not all(isinstance(alias, str) for alias in self.aliases):
            raise ValueError(f"Roster character {self.id}: aliases must be strings")
        object.__setattr__(self, "voice_direction", normalize_voice_direction(self.voice_direction))
        object.__setattr__(self, "merged_ids", tuple(self.merged_ids))
        for merged_id in self.merged_ids:
            if (not isinstance(merged_id, str) or not ROSTER_ID_PATTERN.fullmatch(merged_id)
                    or merged_id in RESERVED_SPEAKER_IDS):
                raise ValueError(f"Roster character {self.id}: invalid merged id {merged_id!r}")

    def name_keys(self) -> set[str]:
        """The normalized name and aliases (see name_key)."""
        return {name_key(self.name), *(name_key(alias) for alias in self.aliases)}

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "aliases": list(self.aliases),
            "gender": self.gender,
            "description": self.description,
            "is_narrator": self.is_narrator,
            "prominence": self.prominence.value,
            "voice_hint": self.voice_hint,
        }
        if self.merged_ids:
            value["merged_ids"] = list(self.merged_ids)
        if self.voice_direction:
            value["voice_direction"] = self.voice_direction
        return value

    @staticmethod
    def from_dict(value: Any) -> RosterCharacter:
        """Parse one stored character. Raises ValueError for a non-object or an
        invalid id or malformed merge history. Other optional fields are
        tolerant (non-string aliases are dropped)."""
        if not isinstance(value, dict):
            raise ValueError(f"Expected a roster character object. Value: {value!r}")
        char_id = value.get("id")
        if not isinstance(char_id, str):
            raise ValueError(f"Roster character without a string id: {value!r}")
        raw_aliases = value.get("aliases")
        aliases = tuple(a for a in raw_aliases if isinstance(a, str)) if isinstance(raw_aliases, list) else ()
        merged_ids = value.get("merged_ids", [])
        if not isinstance(merged_ids, list):
            raise ValueError(f"Roster character {char_id}: merged ids must be a list")
        return RosterCharacter(
            id=char_id,
            name=_string(value.get("name"), char_id),
            aliases=aliases,
            gender=_string(value.get("gender"), "unknown"),
            description=_string(value.get("description"), ""),
            is_narrator=value.get("is_narrator") is True,
            prominence=Prominence.from_value(value.get("prominence")),
            voice_hint=_string(value.get("voice_hint"), ""),
            merged_ids=tuple(merged_ids),
            voice_direction=_string(value.get("voice_direction"), ""),
        )


@dataclass(frozen=True)
class RosterNarration:
    person: NarrativePerson = NarrativePerson.UNKNOWN
    notes: str = ""
    voice_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"person": self.person.value, "notes": self.notes, "voice_hint": self.voice_hint}

    @staticmethod
    def from_dict(value: Any) -> RosterNarration:
        """Tolerant: a missing or malformed value loads as the default."""
        if not isinstance(value, dict):
            return RosterNarration()
        return RosterNarration(
            person=NarrativePerson.from_value(value.get("person")),
            notes=_string(value.get("notes"), ""),
            voice_hint=_string(value.get("voice_hint"), ""),
        )


@dataclass(frozen=True)
class Roster:
    """
    The canonical speaker list for a book, in roster order (most dialog
    first). Character ids are unique and at most one character is the
    narrator (ValueError otherwise).
    """
    characters: tuple[RosterCharacter, ...]
    narration: RosterNarration = field(default_factory=RosterNarration)

    def __post_init__(self) -> None:
        object.__setattr__(self, "characters", tuple(self.characters))
        ids = [character.id for character in self.characters]
        duplicates = sorted({char_id for char_id in ids if ids.count(char_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate roster ids: {', '.join(duplicates)}")
        assigned_ids = set(ids)
        for character in self.characters:
            for merged_id in character.merged_ids:
                if merged_id in assigned_ids:
                    raise ValueError(f"Conflicting merged roster id: {merged_id}")
                assigned_ids.add(merged_id)
        narrators = [character.id for character in self.characters if character.is_narrator]
        if len(narrators) > 1:
            raise ValueError(f"More than one narrator: {', '.join(narrators)}")

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(character.id for character in self.characters)

    @property
    def narrator(self) -> RosterCharacter | None:
        return next((character for character in self.characters if character.is_narrator), None)

    def get(self, char_id: str) -> RosterCharacter | None:
        return next((character for character in self.characters if character.id == char_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "narration": self.narration.to_dict(),
            "characters": [character.to_dict() for character in self.characters],
        }

    @staticmethod
    def from_dict(value: Any) -> Roster:
        """Parse a stored roster (an object with "characters" and optional
        "narration"; other keys are ignored). Raises ValueError for a
        structurally invalid roster or any invalid character."""
        if not isinstance(value, dict) or not isinstance(value.get("characters"), list):
            raise ValueError("Expected a roster object with a 'characters' list")
        return Roster(
            characters=tuple(RosterCharacter.from_dict(item) for item in value["characters"]),
            narration=RosterNarration.from_dict(value.get("narration")),
        )


ROSTER_FILE_FORMAT = "speaker_roster.v1"


@dataclass(frozen=True)
class RosterFile:
    """
    The full-book pass's result: the roster, plus provenance for checking
    whether it still fits the book (`section_text_hashes`, one per book
    section) and for diagnostics (`llm` settings, the LLM's `raw_reply`).

    On disk:

        {
            "format": "speaker_roster.v1",
            "created": "2026-10-22T12:55:07-04:00",
            "llm": {"url": "...", "model": "...", "extra_params": {...}},
            "book_title": "Leech",
            "section_text_hashes": ["9f1c...", ...],
            "roster": {"narration": {...}, "characters": [...]},
            "raw_reply": "..."
        }
    """
    roster: Roster
    created: str = ""
    llm: dict[str, Any] = field(default_factory=dict, hash=False)
    book_title: str = ""
    section_text_hashes: tuple[str, ...] = ()
    raw_reply: str = ""

    def __post_init__(self) -> None:
        # Own copies (callers may pass a list / a dict they keep using).
        object.__setattr__(self, "llm", dict(self.llm))
        object.__setattr__(self, "section_text_hashes", tuple(self.section_text_hashes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ROSTER_FILE_FORMAT,
            "created": self.created,
            "llm": dict(self.llm),
            "book_title": self.book_title,
            "section_text_hashes": list(self.section_text_hashes),
            "roster": self.roster.to_dict(),
            "raw_reply": self.raw_reply,
        }

    @staticmethod
    def from_dict(value: Any) -> RosterFile:
        """Raises ValueError for another format or an invalid roster;
        provenance fields are tolerant (malformed values load as defaults)."""
        if not isinstance(value, dict) or value.get("format") != ROSTER_FILE_FORMAT:
            raise ValueError(f"Not a {ROSTER_FILE_FORMAT} roster file")
        raw_llm = value.get("llm")
        raw_hashes = value.get("section_text_hashes")
        return RosterFile(
            roster=Roster.from_dict(value.get("roster")),
            created=_string(value.get("created"), ""),
            llm=raw_llm if isinstance(raw_llm, dict) else {},
            book_title=_string(value.get("book_title"), ""),
            section_text_hashes=tuple(h for h in raw_hashes if isinstance(h, str))
            if isinstance(raw_hashes, list) else (),
            raw_reply=_string(value.get("raw_reply"), ""),
        )


def _string(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default
