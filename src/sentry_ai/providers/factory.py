"""Resolve a provider name to a VLMProvider instance."""

from sentry_ai.providers.base import VLMProvider
from sentry_ai.providers.minicpm_v import MiniCPMVProvider
from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.providers.qwen3_vl import Qwen3VLProvider
from sentry_ai.providers.qwen_vl import QwenVLProvider
from sentry_ai.providers.vllm_qwen3 import Qwen3VLvLLMProvider

_REGISTRY: dict[str, type] = {
    "qwen3-vl-4b": Qwen3VLProvider,  # DEFAULT — Ollama (ADR-0026 amendment)
    "minicpm-v-2.6": MiniCPMVProvider,  # alt/rollback — Ollama
    "qwen3-vl-vllm": Qwen3VLvLLMProvider,  # scale-path — vLLM on dedicated Linux GPU
    "qwen2.5-vl-7b": QwenVLProvider,  # deprecated — kept for rollback (ADR-0026)
}


def list_provider_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_provider_class(name: str) -> type | None:
    """The provider class (for static attrs like `runtime`/`model_tag`), or None."""
    return _REGISTRY.get(name)


def provider_uses_ollama(name: str | None) -> bool:
    """True when provider ``name`` runs on the Ollama runtime — i.e. Ollama being
    down actually breaks verify. False for vLLM providers (there Ollama is not in the
    verify path, so a red 'Ollama down' dot would be a false alarm). Unknown/unset
    names fall back to the historical Ollama default so a real dependency is never
    hidden; callers should pass an already-resolved provider name."""
    cls = get_provider_class(name) if name else None
    if cls is None:
        return True
    return getattr(cls, "runtime", "ollama") == "ollama"


def resolve_provider_name(requested: str | None) -> str:
    """Pick the effective VLM provider (central control, ADR-0026).

    Priority: explicit per-request `requested` > central value pushed from the
    backend (superadmin dropdown, polled by config_poller) > settings.default_provider
    (.env bootstrap). The central value is ignored unless it's a known provider,
    so a stale/typo'd dashboard value can never break verify.
    """
    if requested:
        return requested
    from sentry_ai.runtime_config import get_central_provider
    from sentry_ai.settings import get_settings

    central = get_central_provider()
    if central and central in _REGISTRY:
        return central
    return get_settings().default_provider


def get_provider(name: str, ollama: OllamaClient) -> VLMProvider:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown provider '{name}'. Known: {sorted(_REGISTRY.keys())}")
    return cls(ollama)  # type: ignore[no-any-return]
