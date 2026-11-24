"""Optional delivery metadata for one spoken quotation."""
from __future__ import annotations

from dataclasses import dataclass
import re


DELIVERY_EVENTS = (
    "laughs", "chuckles", "giggles", "sobs", "whimpers", "groans",
    "sighs", "gasps", "clears throat", "coughs",
)
MAX_DELIVERY_DIRECTION_CHARS = 80
_DIRECTION_PATTERN = re.compile(r"[\w ,.'’\-]+")


@dataclass(frozen=True)
class DeliveryCue:
    """A short performance direction and optional sounds at the quote boundaries."""

    direction: str = ""
    before: str = ""
    after: str = ""

    def __post_init__(self) -> None:
        direction = " ".join(self.direction.split()) if isinstance(self.direction, str) else ""
        if len(direction) > MAX_DELIVERY_DIRECTION_CHARS or not _DIRECTION_PATTERN.fullmatch(direction):
            direction = ""
        object.__setattr__(self, "direction", direction)
        for field in ("before", "after"):
            value = getattr(self, field)
            event = " ".join(value.casefold().split()) if isinstance(value, str) else ""
            object.__setattr__(self, field, event if event in DELIVERY_EVENTS else "")

    def __bool__(self) -> bool:
        return bool(self.direction or self.before or self.after)

    def with_boundaries(self, *, before: bool, after: bool) -> DeliveryCue:
        if (before or not self.before) and (after or not self.after):
            return self
        return DeliveryCue(self.direction, self.before if before else "", self.after if after else "")

    def to_dict(self) -> dict[str, str]:
        return {key: value for key in ("direction", "before", "after") if (value := getattr(self, key))}

    @staticmethod
    def from_dict(value: object) -> DeliveryCue | None:
        if not isinstance(value, dict):
            return None
        return DeliveryCue(value.get("direction", ""), value.get("before", ""), value.get("after", ""))
