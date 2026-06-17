"""Strict Pydantic schema enforcing VLM JSON output shape."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, ValidationError, model_validator


class Category(StrEnum):
    browsing = "browsing"
    cart_pickup = "cart_pickup"
    pocket_conceal = "pocket_conceal"
    bag_conceal = "bag_conceal"
    other = "other"


# Most-severe first. The single `category` (kept for backward compat: alert level,
# analytics, Telegram, RAG) is the highest-severity action detected in the clip.
_SEVERITY_ORDER: tuple[Category, ...] = (
    Category.pocket_conceal,
    Category.bag_conceal,
    Category.cart_pickup,
    Category.browsing,
    Category.other,
)


class VLMOutput(BaseModel):
    """The exact JSON shape VLMs must return.

    Multi-label (ADR pivot 2026-06-17): a single clip can contain several actions
    (e.g. browse then conceal), so the VLM reports every action that occurred in
    `actions`. `category` is the derived primary (most-severe) for the parts of the
    stack that still expect one label.
    """

    # Empty/omitted is coerced to [other] in the validator (cheaper than failing
    # parse → a wasted retry on the slow VLM).
    actions: list[Category] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _sanity(self) -> Self:
        # De-dup while preserving order; an empty list can't pass min_length, but
        # guard anyway so `primary`/`category` never index an empty sequence.
        seen: list[Category] = []
        for a in self.actions:
            if a not in seen:
                seen.append(a)
        self.actions = seen or [Category.other]
        return self

    @property
    def primary(self) -> Category:
        """Highest-severity detected action."""
        for c in _SEVERITY_ORDER:
            if c in self.actions:
                return c
        return self.actions[0]

    @property
    def category(self) -> Category:
        """Backward-compat alias: the primary (most-severe) action."""
        return self.primary


class VLMParseError(Exception):
    """Raised when VLM output cannot be parsed into VLMOutput."""

    def __init__(self, raw: str, cause: ValidationError | Exception):
        self.raw = raw
        self.cause = cause
        super().__init__(f"VLM output invalid: {cause}")
