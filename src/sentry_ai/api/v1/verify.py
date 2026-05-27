"""POST /v1/verify — run Stage 2 inference on a clip."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from sentry_ai.dependencies import OllamaClientDep
from sentry_ai.pipeline.verifier import verify_clip
from sentry_ai.providers.factory import get_provider
from sentry_ai.schemas.verify import VerifyRequest, VerifyResponse
from sentry_ai.settings import get_settings

router = APIRouter(prefix="/v1", tags=["verify"])


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    body: VerifyRequest,
    ollama: Annotated["OllamaClientDep", Depends(OllamaClientDep)],
) -> VerifyResponse:
    settings = get_settings()
    provider_name = body.provider or settings.default_provider

    try:
        provider = get_provider(provider_name, ollama.client)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    clip_path = Path(body.clip_path)
    # Offload blocking stat to a thread to keep the event loop free.
    import asyncio

    exists = await asyncio.to_thread(clip_path.exists)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip not found at {body.clip_path}",
        )

    output, latency_ms, frames_used = await verify_clip(clip_path=clip_path, provider=provider)

    return VerifyResponse(
        category=output.category,
        confidence=output.confidence,
        reasoning=output.reasoning,
        model_name=provider.name,
        inference_latency_ms=latency_ms,
        frames_used=frames_used,
    )
