"""Node diagnostics push — a rich, on-demand snapshot of WHAT is wrong at each
pipeline stage, sent to the backend so the operator reads it off the stage pages
(docs/26 diagnostics console) instead of SSHing into node logs.

This is a SEPARATE outbound channel from the heartbeat (different endpoint,
no schema overlap): the node is behind NAT so push-out is the reliable path.
Best-effort — a diag failure never affects detection/alerts.
"""

from __future__ import annotations

import threading
import time

import httpx

from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.diag")

_PUSH_INTERVAL_SEC = 20.0
_TIMEOUT_SEC = 8.0
_thread: threading.Thread | None = None
_stop = threading.Event()


def collect_diag() -> dict[str, object]:
    """Aggregate the per-stage diagnostics from the live in-process state."""
    from sentry_ai import __version__, runtime_config
    from sentry_ai.live_worker.manager import get_manager

    settings = get_settings()

    # Stage: camera + YOLO (per-worker runtime).
    workers = []
    try:
        for w in get_manager().status():
            workers.append(
                {
                    "camera_id": w.camera_id,
                    "running": w.running,
                    "fps_capture": round(w.fps_capture, 1),
                    "fps_inference": round(w.fps_inference, 1),
                    "frames_total": w.frames_total,
                    "detections_total": w.detections_total,
                    "last_error": w.last_error,
                }
            )
    except Exception:  # noqa: BLE001
        log.warning("diag.workers_failed", exc_info=True)

    # Stage: VLM (verdict history + raw-on-failure + config the verdicts ran with).
    health = runtime_config.get_provider_health()
    frames_per_clip = runtime_config.resolve_frames_per_clip(settings.frames_per_clip)
    frame_max_dim = runtime_config.resolve_frame_max_dim(settings.frame_max_dim)
    verdicts = runtime_config.get_vlm_verdicts()
    parsed_ct = sum(1 for v in verdicts if v.get("parsed"))
    total_ct = len(verdicts)
    vlm = {
        "provider_effective": health.effective,
        "provider_ready": health.ready,
        "provider_error": health.error,
        "breach_mode": runtime_config.resolve_breach_mode(),
        "activity": runtime_config.get_vlm_activity(),
        "config": {
            "num_predict": settings.vlm_num_predict,
            "num_ctx": settings.vlm_num_ctx,
            "frames_per_clip": frames_per_clip,
            "frame_max_dim": frame_max_dim,
            "retry_on_parse_error": settings.retry_on_parse_error,
        },
        "parse_fail_pct": (round(100 * (total_ct - parsed_ct) / total_ct) if total_ct else None),
        "verdicts": verdicts,  # newest-last; each has ts/category/confidence/latency_ms/frames_used/parsed/raw
    }

    return {
        "ts": time.time(),
        "version": __version__,
        "yolo_model": settings.yolo_pose_weights.removesuffix(".pt"),
        "workers": workers,
        "vlm": vlm,
    }


def _push_once(client: httpx.Client) -> None:
    settings = get_settings()
    if not settings.sentry_backend_url or not settings.sentry_backend_service_token:
        return
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/internal/node-diag"
    headers = {
        "Authorization": f"Bearer {settings.sentry_backend_service_token}",
        "Content-Type": "application/json",
    }
    try:
        r = client.post(url, json=collect_diag(), headers=headers)
        if r.status_code >= 400:
            log.warning("diag.push_rejected", status=r.status_code, body=r.text[:200])
    except httpx.HTTPError as e:
        log.warning("diag.push_error", error=str(e))


def _loop() -> None:
    with httpx.Client(timeout=_TIMEOUT_SEC) as client:
        while not _stop.is_set():
            _push_once(client)
            _stop.wait(_PUSH_INTERVAL_SEC)


def start_diag() -> None:
    """Spawn the diag push thread. No-op if unpaired or already running."""
    global _thread
    settings = get_settings()
    if not settings.sentry_backend_url or not settings.sentry_backend_service_token:
        log.info("diag.disabled", reason="not paired")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="diag-push", daemon=True)
    _thread.start()
    log.info("diag.started", interval_sec=_PUSH_INTERVAL_SEC)


def stop_diag() -> None:
    _stop.set()
