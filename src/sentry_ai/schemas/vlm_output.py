"""Strict Pydantic schema enforcing VLM JSON output shape."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, ValidationError, model_validator


class Category(StrEnum):
    browsing = "browsing"
    cart_pickup = "cart_pickup"
    pocket_conceal = "pocket_conceal"
    other = "other"


class VLMOutput(BaseModel):
    """The exact JSON shape VLMs must return."""

    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _sanity(self) -> Self:
        return self


class VLMParseError(Exception):
    """Raised when VLM output cannot be parsed into VLMOutput."""

    def __init__(self, raw: str, cause: ValidationError | Exception):
        self.raw = raw
        self.cause = cause
        super().__init__(f"VLM output invalid: {cause}")
