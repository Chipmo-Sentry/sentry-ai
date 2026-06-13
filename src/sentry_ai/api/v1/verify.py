"""POST /v1/verify — run Stage 2 inference on a clip."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from sentry_ai.auth import require_service_token
from sentry_ai.dependencies import OllamaClientDep
from sentry_ai.pipeline.verifier import verify_clip
from sentry_ai.providers.factory import get_provider, resolve_provider_name
from sentry_ai.schemas.verify import (
    CutVerifyRequest,
    CutVerifyResponse,
    VerifyRequest,
    VerifyResponse,
)
from sentry_ai.settings import get_settings

router = APIRouter(prefix="/v1", tags=["verify"], dependencies=[Depends(require_service_token)])


def _resolve_clip_path(raw: str) -> Path:
    """Resolve a request clip_path, confining it to CLIP_STORAGE_ROOT when set.

    Blocks the arbitrary-host-file read: '../' traversal and absolute paths
    outside the configured storage root are rejected with 400.
    """
    settings = get_settings()
    candidate = Path(raw).resolve()
    root = settings.clip_storage_root
    if root:
        root_resolved = Path(root).resolve()
        if not candidate.is_relative_to(root_resolved):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="clip_path is outside the allowed storage root",
            )
    return candidate


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    body: VerifyRequest,
    ollama: Annotated["OllamaClientDep", Depends(OllamaClientDep)],
) -> VerifyResponse:
    provider_name = resolve_provider_name(body.provider)

    try:
        provider = get_provider(provider_name, ollama.client)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    clip_path = _resolve_clip_path(body.clip_path)
    # Offload blocking stat to a thread to keep the event loop free.
    import asyncio

    exists = await asyncio.to_thread(clip_path.exists)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip not found at {body.clip_path}",
        )

    # RAG (docs/19 Phase 4): if the caller described the event, fetch similar
    # past staff-verified cases for this store and feed them to the VLM.
    store_context = None
    if body.rag_query:
        from sentry_ai import rag

        store_context = await rag.retrieve_context(
            str(body.store_id) if body.store_id else None, body.rag_query
        )

    output, latency_ms, frames_used = await verify_clip(
        clip_path=clip_path, provider=provider, store_context=store_context
    )

    # RAG (docs/19 Phase 4): embed the reasoning so the backend can store it on
    # the alert → staff feedback later spawns a retrievable verified_case.
    from sentry_ai import rag

    embedding = await rag.embed_text(output.reasoning)

    return VerifyResponse(
        category=output.category,
        confidence=output.confidence,
        reasoning=output.reasoning,
        model_name=provider.name,
        inference_latency_ms=latency_ms,
        frames_used=frames_used,
        embedding=embedding,
    )


@router.post("/cut-verify", response_model=CutVerifyResponse)
async def cut_verify(
    body: CutVerifyRequest,
    ollama: Annotated["OllamaClientDep", Depends(OllamaClientDep)],
) -> CutVerifyResponse:
    """Live-breach path (docs/19 I5): cut a clip from THIS node's MediaMTX
    recordings, verify it (VLM + RAG), and return the verdict + the clip bytes
    (base64) so the Railway backend — which can't read these recordings — can
    store the clip and create the alert.
    """
    import base64

    from sentry_ai import clip_cutter, rag

    provider_name = resolve_provider_name(body.provider)
    try:
        provider = get_provider(provider_name, ollama.client)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    # 1. Cut the breach window from local recordings.
    try:
        cut = await clip_cutter.cut_window(
            body.mediamtx_path,
            start_offset_sec=body.start_offset_sec,
            duration_sec=body.duration_sec,
        )
    except clip_cutter.ClipCutError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"clip cut failed: {e}"
        ) from e

    # 2. RAG few-shot (optional) + verify.
    store_context = None
    if body.rag_query:
        store_context = await rag.retrieve_context(
            str(body.store_id) if body.store_id else None, body.rag_query
        )
    output, latency_ms, frames_used = await verify_clip(
        clip_path=Path(cut.storage_path), provider=provider, store_context=store_context
    )
    embedding = await rag.embed_text(output.reasoning)

    # 3. Read clip bytes, then clean up the temp file. Small (~1-3 MB) sync I/O
    # in a per-breach handler — fine to not offload.
    clip_bytes = Path(cut.storage_path).read_bytes()  # noqa: ASYNC240
    Path(cut.storage_path).unlink(missing_ok=True)  # noqa: ASYNC240

    return CutVerifyResponse(
        category=output.category,
        confidence=output.confidence,
        reasoning=output.reasoning,
        model_name=provider.name,
        inference_latency_ms=latency_ms,
        frames_used=frames_used,
        embedding=embedding,
        duration_sec_clip=cut.duration_sec,
        captured_at=cut.captured_at.isoformat(),
        file_size_bytes=cut.file_size_bytes,
        clip_b64=base64.b64encode(clip_bytes).decode("ascii"),
    )
