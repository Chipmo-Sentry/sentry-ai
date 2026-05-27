"""Resolve a provider name to a VLMProvider instance."""

from sentry_ai.providers.base import VLMProvider
from sentry_ai.providers.minicpm_v import MiniCPMVProvider
from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.providers.qwen_vl import QwenVLProvider

_REGISTRY: dict[str, type] = {
    "minicpm-v-2.6": MiniCPMVProvider,
    "qwen2.5-vl-7b": QwenVLProvider,
}


def list_provider_names() -> list[str]:
    return list(_REGISTRY.keys())


def get_provider(name: str, ollama: OllamaClient) -> VLMProvider:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown provider '{name}'. Known: {sorted(_REGISTRY.keys())}")
    return cls(ollama)  # type: ignore[no-any-return]
