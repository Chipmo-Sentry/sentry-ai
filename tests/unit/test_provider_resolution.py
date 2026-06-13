"""Tests for central-control provider resolution (ADR-0026).

Priority: explicit request > central (backend-pushed, if known) > .env default.
"""

import pytest

from sentry_ai import runtime_config
from sentry_ai.providers.factory import resolve_provider_name
from sentry_ai.settings import get_settings


@pytest.fixture(autouse=True)
def _reset() -> None:
    runtime_config.set_central_provider(None)
    get_settings.cache_clear()
    yield
    runtime_config.set_central_provider(None)
    get_settings.cache_clear()


def test_explicit_request_wins_over_central() -> None:
    runtime_config.set_central_provider("minicpm-v-2.6")
    assert resolve_provider_name("qwen3-vl-vllm") == "qwen3-vl-vllm"


def test_central_used_when_no_request() -> None:
    runtime_config.set_central_provider("minicpm-v-2.6")
    assert resolve_provider_name(None) == "minicpm-v-2.6"


def test_unknown_central_falls_back_to_env_default() -> None:
    runtime_config.set_central_provider("totally-made-up")
    # falls back to settings.default_provider (the .env default)
    assert resolve_provider_name(None) == get_settings().default_provider


def test_no_central_uses_env_default() -> None:
    assert resolve_provider_name(None) == get_settings().default_provider
