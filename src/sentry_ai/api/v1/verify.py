"""POST /v1/verify — run Stage 2 inference on a clip."""

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

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

# --- Edge clip-upload GPU gate (ADR-0029 §12 / I5) ---
# A single GPU can't run N VLM verifies at once without risking VRAM OOM, so
# edge-clip verifies are serialised through a semaphore (size = max_concurrency)
# and the queue is bounded — once `_edge_inflight` (running + waiting) hits
# concurrency + max_queue we shed with 503 instead of piling up. The semaphore
# is created lazily so it binds to the running event loop.
_edge_gate: asyncio.Semaphore | None = None
_edge_inflight = 0


def _get_edge_gate() -> asyncio.Semaphore:
    global _edge_gate
    if _edge_gate is None:
        _edge_gate = asyncio.Semaphore(get_settings().edge_verify_max_concurrency)
    return _edge_gate


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


@router.post("/edge-clip-upload", response_model=VerifyResponse)
async def edge_clip_upload(
    ollama: Annotated["OllamaClientDep", Depends(OllamaClientDep)],
    clip: Annotated[UploadFile, File()],
    store_id: Annotated[str | None, Form()] = None,
    provider: Annotated[str | None, Form()] = None,
    rag_query: Annotated[str | None, Form()] = None,
) -> VerifyResponse:
    """Edge Stage-1 path (ADR-0029): the store agent (sentry-agent-pc) ran
    YOLO + behaviour LOCALLY and uploads only the suspicious clip BYTES. The
    Railway backend forwards them here as multipart — `/v1/verify`'s clip_path
    is read on THIS host, so it can't work when the backend and the GPU node are
    different machines. We verify the bytes and return the same verdict shape;
    the backend owns tenancy + behaviour data and builds the alert.
    """
    import os
    import tempfile

    from sentry_ai import rag

    global _edge_inflight
    settings = get_settings()
    # Backpressure (I5): shed load BEFORE buffering when the node is saturated,
    # so a flood of uploads can't pile up unbounded pending verifies.
    if _edge_inflight >= settings.edge_verify_max_concurrency + settings.edge_verify_max_queue:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI node busy — retry shortly.",
        )
    _edge_inflight += 1
    try:
        provider_name = resolve_provider_name(provider)
        try:
            prov = get_provider(provider_name, ollama.client)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

        # Stream the upload to a temp file, confined under CLIP_STORAGE_ROOT when
        # set (the same boundary /v1/verify enforces). Reject (413, P3) the moment
        # the byte count exceeds the cap so a huge body can't buffer in RAM (the
        # old code did one `await clip.read()`). Always cleaned up in `finally`.
        root = settings.clip_storage_root
        tmp_dir: str | None = None
        if root:
            await asyncio.to_thread(os.makedirs, root, exist_ok=True)
            tmp_dir = root

        max_bytes = settings.edge_clip_max_mb * 1024 * 1024
        fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=tmp_dir)
        tmp_path = Path(tmp_name)
        try:
            written = 0
            too_large = False
            with os.fdopen(fd, "wb") as fh:
                while chunk := await clip.read(1 << 20):
                    written += len(chunk)
                    if written > max_bytes:
                        too_large = True
                        break
                    await asyncio.to_thread(fh.write, chunk)
            if too_large:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Клип хэт том байна (дээд хязгаар {settings.edge_clip_max_mb}MB).",
                )

            store_context = None
            if rag_query:
                store_context = await rag.retrieve_context(store_id, rag_query)

            # GPU gate (I5): serialise the VLM verify so concurrent uploads don't
            # co-run on one GPU and OOM. Other requests wait here (bounded above).
            async with _get_edge_gate():
                output, latency_ms, frames_used = await verify_clip(
                    clip_path=tmp_path, provider=prov, store_context=store_context
                )
            embedding = await rag.embed_text(output.reasoning)
            return VerifyResponse(
                category=output.category,
                confidence=output.confidence,
                reasoning=output.reasoning,
                model_name=prov.name,
                inference_latency_ms=latency_ms,
                frames_used=frames_used,
                embedding=embedding,
            )
        finally:
            await asyncio.to_thread(tmp_path.unlink, missing_ok=True)
    finally:
        _edge_inflight -= 1
