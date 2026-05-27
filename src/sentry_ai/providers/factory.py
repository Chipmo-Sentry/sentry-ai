"""Resolve a provider name to a VLMProvider instance."""

from sentry_ai.providers.base import VLMProvider
from sentry_ai.providers.minicpm_v import MiniCPMVProvider
from sentry_ai.providers.ollama_client import OllamaClient

_REGISTRY: dict[str, type] = {
    "minicpm-v-2.6": MiniCPMVProvider,
    # Qwen2.5-VL will register here in Session 2
}


def list_provider_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_provider(name: str, ollama: OllamaClient) -> VLMProvider:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown provider '{name}'. Known: {sorted(_REGISTRY.keys())}")
    return cls(ollama)  # type: ignore[no-any-return]
