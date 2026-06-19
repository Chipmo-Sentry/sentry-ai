"""Unit tests for VLMOutput strict validation."""

import json

import pytest
from pydantic import ValidationError

from sentry_ai.schemas.vlm_output import Category, VLMOutput, VLMParseError


def test_valid_output_parses() -> None:
    # Multi-label contract (ADR 2026-06-17): the VLM reports every action; `category`
    # is the derived most-severe one (pocket_conceal outranks browsing).
    raw = json.dumps(
        {
            "actions": ["browsing", "pocket_conceal"],
            "confidence": 0.82,
            "reasoning": "Хүн бараа авч халаасанд хийсэн, эргэж харсан.",
        }
    )
    out = VLMOutput.model_validate_json(raw)
    assert out.actions == [Category.browsing, Category.pocket_conceal]
    assert out.primary is Category.pocket_conceal
    assert out.category is Category.pocket_conceal
    assert 0.0 <= out.confidence <= 1.0


def test_unknown_category_rejected() -> None:
    raw = json.dumps({"actions": ["ALARM"], "confidence": 0.5, "reasoning": "x" * 50})
    with pytest.raises(ValidationError):
        VLMOutput.model_validate_json(raw)


def test_confidence_out_of_range_rejected() -> None:
    raw = json.dumps({"category": "browsing", "confidence": 1.5, "reasoning": "x" * 50})
    with pytest.raises(ValidationError):
        VLMOutput.model_validate_json(raw)


def test_empty_reasoning_rejected() -> None:
    raw = json.dumps({"category": "other", "confidence": 0.1, "reasoning": ""})
    with pytest.raises(ValidationError):
        VLMOutput.model_validate_json(raw)


def test_parse_error_wraps_cause() -> None:
    cause = ValueError("nope")
    err = VLMParseError("garbage", cause)
    assert err.raw == "garbage"
    assert err.cause is cause
