"""GET /healthz + GET /v1/models — operational endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends

from sentry_ai import __version__
from sentry_ai.dependencies import OllamaClientDep
from sentry_ai.providers.factory import list_provider_names
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
    return HealthResponse(
        status="ok" if reachable else "degraded",
        version=__version__,
        ollama_reachable=reachable,
        loaded_models=models,
    )


@router.get("/v1/models", response_model=list[str])
async def models() -> list[str]:
    """List provider names this sentry-ai supports (not the Ollama-installed ones)."""
    return list_provider_names()
