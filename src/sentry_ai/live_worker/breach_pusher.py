"""Node-push live alerts.

When a tracked person's risk crosses the alert threshold and SUSTAINS, the
camera worker calls submit_breach(). We then — on a single background worker so
only ONE cut+VLM runs at a time (one GPU / one Ollama) — cut the breach clip
from THIS node's MediaMTX recordings, run the VLM, apply the ignore gate, and
POST the finished alert to the backend's /api/v1/internal/live-alert.

This is the reliable direction (node → cloud, same as live metadata). It
replaces the fragile cloud → node /v1/cut-verify pull that broke whenever the
node wasn't reachable from the cloud (NAT / tunnel down).

Each breach runs in its own short-lived event loop (asyncio.run) with a FRESH
Ollama client, so there's no cross-loop client reuse with the FastAPI app loop.
The single-worker pool serialises breaches; the camera frame loop only submits.
"""

from __future__ import annotations

import asyncio
import base64
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.live_worker.breach_pusher")

# One worker → at most one cut+VLM at a time (the node has a single GPU/Ollama).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="breach")
_POST_TIMEOUT_SEC = 15.0
# Mirror of the backend's derive_alert_level "ignore" gate so the node never
# pushes a verdict the backend would just drop.
_IGNORE_CONF = 0.30


def submit_breach(
    *,
    mediamtx_path: str,
    person_id: int,
    peak_risk_pct: float,
    behaviors: list[str],
    sequences: list[str],
    behavior_detail: list[dict[str, Any]],
    episode_started_ms: int | None,
    breach_ts_ms: int,
) -> None:
    """Fire-and-forget: queue a breach for cut+VLM+push. Never blocks the caller."""
    if not get_settings().live_alert_push_enabled:
        return
    _executor.submit(
        _handle,
        mediamtx_path,
        person_id,
        peak_risk_pct,
        behaviors,
        sequences,
        behavior_detail,
        episode_started_ms,
        breach_ts_ms,
    )


def _handle(
    mediamtx_path: str,
    person_id: int,
    peak_risk_pct: float,
    behaviors: list[str],
    sequences: list[str],
    behavior_detail: list[dict[str, Any]],
    episode_started_ms: int | None,
    breach_ts_ms: int,
) -> None:
    try:
        asyncio.run(
            _cut_verify_push(
                mediamtx_path,
                person_id,
                peak_risk_pct,
                behaviors,
                sequences,
                behavior_detail,
                episode_started_ms,
                breach_ts_ms,
            )
        )
    except Exception:  # noqa: BLE001 — a breach failure must never kill the worker
        log.exception("breach_push.failed", mediamtx_path=mediamtx_path, person_id=person_id)


def _clip_window(
    *,
    episode_started_ms: int | None,
    breach_ts_ms: int,
    pre_pad: int,
    post_roll: float,
    max_sec: int,
) -> tuple[int, int]:
    """(start_offset_sec, duration_sec) covering the whole episode. Mirrors the
    backend compute_clip_window — we sleep post_roll first, so node-"now" at cut
    time ≈ breach + post_roll."""
    if episode_started_ms is not None and episode_started_ms <= breach_ts_ms:
        pre_sec = (breach_ts_ms - episode_started_ms) / 1000.0 + pre_pad
    else:
        pre_sec = 5.0
    duration = math.ceil(pre_sec + post_roll + 2.0)
    duration = max(5, min(max_sec, duration))
    return -duration, duration


async def _cut_verify_push(
    mediamtx_path: str,
    person_id: int,
    peak_risk_pct: float,
    behaviors: list[str],
    sequences: list[str],
    behavior_detail: list[dict[str, Any]],
    episode_started_ms: int | None,
    breach_ts_ms: int,
) -> None:
    from sentry_ai import clip_cutter, rag
    from sentry_ai.pipeline.verifier import verify_clip
    from sentry_ai.providers.factory import get_provider, resolve_provider_name
    from sentry_ai.providers.ollama_client import OllamaClient

    settings = get_settings()
    post_roll = max(0.0, settings.live_breach_post_roll_sec)
    if post_roll:
        await asyncio.sleep(post_roll)

    start_offset, duration = _clip_window(
        episode_started_ms=episode_started_ms,
        breach_ts_ms=breach_ts_ms,
        pre_pad=settings.live_clip_pre_pad_sec,
        post_roll=post_roll,
        max_sec=settings.live_clip_max_sec,
    )

    try:
        cut = await clip_cutter.cut_window(
            mediamtx_path, start_offset_sec=start_offset, duration_sec=duration
        )
    except clip_cutter.ClipCutError as e:
        log.warning("breach_push.cut_failed", mediamtx_path=mediamtx_path, error=str(e))
        return

    # Fresh Ollama client bound to THIS event loop (avoids cross-loop reuse with
    # the FastAPI app's client). Closed in finally.
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_sec=settings.inference_timeout_sec,
        num_ctx=settings.vlm_num_ctx,
    )
    try:
        provider = get_provider(resolve_provider_name(None), client)
        output, latency_ms, _frames = await verify_clip(
            clip_path=Path(cut.storage_path), provider=provider, store_context=None
        )
    finally:
        await client.aclose()

    # Ignore gate (mirror backend derive_alert_level): browsing / low confidence
    # never becomes an alert — drop it here so we don't ship a clip for nothing.
    if output.category.value == "browsing" or output.confidence < _IGNORE_CONF:
        log.info(
            "breach_push.vlm_cleared",
            mediamtx_path=mediamtx_path,
            person_id=person_id,
            category=output.category.value,
            confidence=output.confidence,
        )
        Path(cut.storage_path).unlink(missing_ok=True)  # noqa: ASYNC240
        return

    # Embedding is best-effort (RAG loop enrichment); never block the alert on it.
    embedding: list[float] | None = None
    try:
        embedding = await rag.embed_text(output.reasoning)
    except Exception:  # noqa: BLE001
        log.warning("breach_push.embed_failed", exc_info=True)

    clip_bytes = Path(cut.storage_path).read_bytes()  # noqa: ASYNC240
    Path(cut.storage_path).unlink(missing_ok=True)  # noqa: ASYNC240

    payload: dict[str, Any] = {
        "camera_id": mediamtx_path,
        "category": output.category.value,
        "confidence": output.confidence,
        "reasoning": output.reasoning,
        "model_name": provider.name,
        "inference_latency_ms": latency_ms,
        "clip_b64": base64.b64encode(clip_bytes).decode("ascii"),
        "file_size_bytes": cut.file_size_bytes,
        "duration_sec_clip": cut.duration_sec,
        "captured_at": cut.captured_at.isoformat(),
        "embedding": embedding,
        "person_id": person_id,
        "peak_risk_pct": peak_risk_pct,
        "triggered_behaviors": behaviors or None,
        "triggered_sequences": sequences or None,
        "triggered_behavior_detail": behavior_detail or None,
    }
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/internal/live-alert"
    headers = {
        "Authorization": f"Bearer {settings.sentry_backend_service_token}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=_POST_TIMEOUT_SEC) as client_http:
            r = await client_http.post(url, json=payload, headers=headers)
            if r.status_code >= 400:
                log.warning("breach_push.post_failed", status=r.status_code, body=r.text[:200])
            else:
                log.info(
                    "breach_push.posted",
                    mediamtx_path=mediamtx_path,
                    person_id=person_id,
                    category=output.category.value,
                    confidence=output.confidence,
                )
    except httpx.HTTPError:
        log.warning("breach_push.post_error", mediamtx_path=mediamtx_path, exc_info=True)
