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


def get_provider(name: str, ollama: OllamaClient) -> VLMProvider:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown provider '{name}'. Known: {sorted(_REGISTRY.keys())}")
    return cls(ollama)  # type: ignore[no-any-return]
