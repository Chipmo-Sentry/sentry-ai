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
import threading
import time
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
# VLM-primary scans fire on a fast cadence (~3s) but the single GPU takes longer
# per clip — drop ticks while one is in flight instead of queueing them up
# unboundedly. This throttles the scan rate to the GPU's real throughput.
_busy_lock = threading.Lock()
_busy = False
# Mirror of the backend's evidence gate so the node only pushes clips that the
# operator actually wants retained: theft or attempted theft, not browsing/cart.
_THEFT_CATEGORIES = {"pocket_conceal", "bag_conceal"}
_THEFT_SAVE_CONF = 0.50


def _should_push_theft_clip(output: Any) -> bool:
    return output.category.value in _THEFT_CATEGORIES and output.confidence >= _THEFT_SAVE_CONF


# Mongolian labels for the behavior-engine signals, so the VLM hint reads as
# plain analyst language ("гар цээж рүү") instead of raw keys.
_BEHAVIOR_LABELS_MN: dict[str, str] = {
    "looking_around": "эргэн тойрноо ажиглах",
    "loitering": "удаан зогсох",
    "rapid_movement": "хурдан хөдөлгөөн",
    "item_pickup": "бараа авах",
    "body_block": "биеэрээ халхлах",
    "crouch": "бөхийх",
    "wrist_to_torso": "гар цээж/бэлхүүс рүү",
    "bag_interaction": "цүнхтэй харьцах",
    "pocket_interaction": "халаастай харьцах",
    "repeated_shelf_visit": "нэг тавиур руу дахин дахин очих",
    "exit_after_concealment": "нуусны дараа гарц руу явах",
}


def _behavior_hint_mn(behaviors: list[str], peak_risk_pct: float) -> str | None:
    """Short Mongolian summary of the edge signals for the VLM prompt, or None
    when there's nothing specific to say (a bare risk number isn't a hint)."""
    labels = [_BEHAVIOR_LABELS_MN.get(b, b) for b in behaviors[:4]]
    if not labels:
        return None
    return f"{', '.join(labels)} (эрсдэл {round(peak_risk_pct)}%)"


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
    pose_sequence: list[dict[str, Any]] | None = None,
) -> None:
    """Fire-and-forget: queue a breach for cut+VLM+push. Never blocks the caller."""
    # Topology is backend-authoritative (ADR-0026): the node pushes only when the
    # centrally-chosen breach_mode is "node_push" (env is just the pre-poll
    # bootstrap). "off" → AI/overlay keep running, but we create no alerts.
    from sentry_ai.runtime_config import resolve_breach_mode

    if resolve_breach_mode() != "node_push":
        return
    global _busy
    with _busy_lock:
        if _busy:
            return  # a scan is still cutting+VLM on the single GPU — skip this tick
        _busy = True
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
        pose_sequence,
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
    pose_sequence: list[dict[str, Any]] | None = None,
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
                pose_sequence,
            )
        )
    except Exception:  # noqa: BLE001 — a breach failure must never kill the worker
        log.exception("breach_push.failed", mediamtx_path=mediamtx_path, person_id=person_id)
    finally:
        global _busy
        with _busy_lock:
            _busy = False


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


async def _report_cleared(
    *,
    mediamtx_path: str,
    reason: str,
    peak_risk_pct: float,
    person_id: int,
    behaviors: list[str],
    category: str | None = None,
    confidence: float | None = None,
) -> None:
    """Tell the backend a breach was detected but produced NO alert (VLM cleared
    it, or the clip-cut failed) so the miss is VISIBLE on the activity timeline
    instead of silently lost. Best-effort; never raises."""
    settings = get_settings()
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/internal/breach-cleared"
    headers = {
        "Authorization": f"Bearer {settings.sentry_backend_service_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "camera_id": mediamtx_path,
        "reason": reason,
        "peak_risk_pct": peak_risk_pct,
        "person_id": person_id,
        "category": category,
        "confidence": confidence,
        "triggered_behaviors": behaviors or None,
    }
    try:
        async with httpx.AsyncClient(timeout=_POST_TIMEOUT_SEC) as client_http:
            await client_http.post(url, json=payload, headers=headers)
    except httpx.HTTPError:
        log.warning("breach_cleared.post_error", mediamtx_path=mediamtx_path, exc_info=True)


async def _cut_verify_push(
    mediamtx_path: str,
    person_id: int,
    peak_risk_pct: float,
    behaviors: list[str],
    sequences: list[str],
    behavior_detail: list[dict[str, Any]],
    episode_started_ms: int | None,
    breach_ts_ms: int,
    pose_sequence: list[dict[str, Any]] | None = None,
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
        await _report_cleared(
            mediamtx_path=mediamtx_path,
            reason="cut_failed",
            peak_risk_pct=peak_risk_pct,
            person_id=person_id,
            behaviors=behaviors,
        )
        return

    # Fresh Ollama client bound to THIS event loop (avoids cross-loop reuse with
    # the FastAPI app's client). Closed in finally.
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_sec=settings.inference_timeout_sec,
        num_ctx=settings.vlm_num_ctx,
        num_predict=settings.vlm_num_predict,
    )
    _verify_t0 = time.monotonic()
    try:
        provider = get_provider(resolve_provider_name(None), client)
        output, latency_ms, _frames = await verify_clip(
            clip_path=Path(cut.storage_path),
            provider=provider,
            store_context=None,
            behavior_hint=_behavior_hint_mn(behaviors, peak_risk_pct),
        )
    except Exception as e:  # noqa: BLE001 — docs/33 P0-5: a VLM outage must not orphan
        # Transport/unexpected VLM failure (verify_clip already retried a cold
        # load once with an extended timeout). Pre-fix this propagated to
        # _handle's catch-all: the cut clip stayed on disk forever (temp-dir
        # fill during any VLM outage) and the miss was invisible. Delete the
        # clip + surface the miss on the activity timeline as a cleared breach.
        log.warning(
            "breach_push.vlm_error",
            mediamtx_path=mediamtx_path,
            person_id=person_id,
            error=str(e)[:200],
        )
        Path(cut.storage_path).unlink(missing_ok=True)  # noqa: ASYNC240
        await _report_cleared(
            mediamtx_path=mediamtx_path,
            reason="vlm_error",
            peak_risk_pct=peak_risk_pct,
            person_id=person_id,
            behaviors=behaviors,
        )
        return
    finally:
        await client.aclose()
    # Split the scan latency so we can SEE where the time goes: total verify wall
    # time vs the VLM's own inference (latency_ms). extract ≈ total − VLM.
    verify_total_ms = int((time.monotonic() - _verify_t0) * 1000)
    log.info(
        "breach_push.verify_timing",
        mediamtx_path=mediamtx_path,
        verify_total_ms=verify_total_ms,
        vlm_ms=latency_ms,
        extract_ms=max(0, verify_total_ms - latency_ms),
        category=output.category.value,
        confidence=round(output.confidence, 2),
    )

    # Only retained clips should be theft/attempt evidence. For benign or
    # unclear breaches, report a cleared episode and delete the local cut.
    if not _should_push_theft_clip(output):
        log.info(
            "breach_push.cleared_by_vlm",
            mediamtx_path=mediamtx_path,
            person_id=person_id,
            category=output.category.value,
            confidence=output.confidence,
        )
        Path(cut.storage_path).unlink(missing_ok=True)  # noqa: ASYNC240
        await _report_cleared(
            mediamtx_path=mediamtx_path,
            reason="vlm_cleared",
            peak_risk_pct=peak_risk_pct,
            person_id=person_id,
            behaviors=behaviors,
            category=output.category.value,
            confidence=output.confidence,
        )
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
        # Multi-label: every action detected in the clip. `category` stays as the
        # primary (most-severe) for backward compat (level/analytics/Telegram/RAG).
        "actions": [a.value for a in output.actions],
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
        # Фаз 0 (ADR-0030): the breaching person's skeleton trajectory → training
        # data for the skeleton-anomaly model once staff verify this alert.
        "pose_sequence": pose_sequence or None,
    }
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/internal/live-alert"
    headers = {
        "Authorization": f"Bearer {settings.sentry_backend_service_token}",
        "Content-Type": "application/json",
    }
    # docs/33 P0-5: the alert POST used to be single-shot — a transient backend
    # blip / slow uplink dropped the ALERT (verdict + evidence clip) forever.
    # Retry once after a short backoff; retry 5xx too (4xx is a permanent
    # rejection, don't repeat it). A doubled timeout on the retry gives a slow
    # store uplink room to move the multi-MB clip payload.
    for attempt in (0, 1):
        try:
            timeout = _POST_TIMEOUT_SEC * (2 if attempt else 1)
            async with httpx.AsyncClient(timeout=timeout) as client_http:
                r = await client_http.post(url, json=payload, headers=headers)
            if r.status_code < 400:
                log.info(
                    "breach_push.posted",
                    mediamtx_path=mediamtx_path,
                    person_id=person_id,
                    category=output.category.value,
                    confidence=output.confidence,
                )
                return
            log.warning("breach_push.post_failed", status=r.status_code, body=r.text[:200])
            if r.status_code < 500:
                return  # permanent rejection — retrying can't help
        except httpx.HTTPError:
            log.warning(
                "breach_push.post_error",
                mediamtx_path=mediamtx_path,
                attempt=attempt,
                exc_info=True,
            )
        if attempt == 0:
            await asyncio.sleep(5.0)
    log.error(
        "breach_push.post_dropped",  # both attempts failed — alert + clip LOST
        mediamtx_path=mediamtx_path,
        person_id=person_id,
        category=output.category.value,
    )
