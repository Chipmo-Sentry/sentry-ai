"""FastAPI dependency wrappers."""

from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.settings import get_settings

_client: OllamaClient | None = None


def _get_client() -> OllamaClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OllamaClient(
            base_url=settings.ollama_base_url,
            timeout_sec=settings.inference_timeout_sec,
            num_ctx=settings.vlm_num_ctx,
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class OllamaClientDep:
    """Dependency that exposes a process-wide OllamaClient."""

    def __init__(self) -> None:
        self.client = _get_client()
