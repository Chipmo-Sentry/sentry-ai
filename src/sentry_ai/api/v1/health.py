"""GET /healthz + GET /v1/models — operational endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends

from sentry_ai import __version__
from sentry_ai.dependencies import OllamaClientDep
from sentry_ai.providers.factory import (
    list_provider_names,
    provider_uses_ollama,
    resolve_provider_name,
)
from sentry_ai.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz(
    ollama: Annotated["OllamaClientDep", Depends(OllamaClientDep)],
) -> HealthResponse:
    reachable = await ollama.client.is_reachable()
    models: list[str] = []
    if reachable:
        try:
            models = await ollama.client.list_models()
        except Exception:  # noqa: BLE001
            models = []
    # Ollama only gates health when the effective VLM provider runs on it. With a
    # vLLM provider active, Ollama being unreachable doesn't degrade verify, so we
    # stay "ok" rather than crying "degraded" over a dependency we don't use.
    ollama_required = provider_uses_ollama(resolve_provider_name(None))
    healthy = reachable or not ollama_required
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        ollama_reachable=reachable,
        ollama_required=ollama_required,
        loaded_models=models,
    )


@router.get("/v1/models", response_model=list[str])
async def models() -> list[str]:
    """List provider names this sentry-ai supports (not the Ollama-installed ones)."""
    return list_provider_names()
