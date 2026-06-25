"""Tests for central-control provider resolution (ADR-0026).

Priority: explicit request > central (backend-pushed, if known) > .env default.
"""

import pytest

from sentry_ai import runtime_config
from sentry_ai.providers.factory import provider_uses_ollama, resolve_provider_name
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


def test_ollama_provider_requires_ollama() -> None:
    # Ollama-runtime providers (the default + alt) genuinely need Ollama.
    assert provider_uses_ollama("qwen3-vl-4b") is True
    assert provider_uses_ollama("minicpm-v-2.6") is True


def test_vllm_provider_does_not_require_ollama() -> None:
    # vLLM provider runs outside Ollama → a down Ollama is not a fault.
    assert provider_uses_ollama("qwen3-vl-vllm") is False


def test_unknown_or_unset_provider_assumes_ollama() -> None:
    # Unknown/unset → conservative default so a real dependency is never hidden.
    assert provider_uses_ollama("totally-made-up") is True
    assert provider_uses_ollama(None) is True
