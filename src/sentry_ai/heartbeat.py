"""AI-node heartbeat — reports telemetry to the backend + polls config.

Runs only when this node has been paired (settings.ai_node_id is set). Posts
to /api/v1/ai-nodes/heartbeat every `heartbeat_interval_sec` using the node's
ai_node JWT (settings.sentry_backend_service_token). Best-effort: failures are
logged, never fatal. The response carries the central config (enabled,
provider, frame_skip) which we expose for new workers to read.

Runs in its OWN daemon THREAD with a synchronous httpx client (NOT an asyncio
task on the FastAPI event loop). The live workers are CPU/GPU-heavy threads; an
asyncio heartbeat on the main loop gets starved under load and stops beating —
which made the node flap to "offline" in superadmin even while it was healthy.
A dedicated thread (same pattern as the metadata emitter) beats reliably.
"""

from __future__ import annotations

import threading

import httpx

from sentry_ai.live_worker import get_manager
from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.heartbeat")

# Latest config pushed from the backend (read by the manager when starting
# workers). None until the first successful heartbeat.
current_config: dict[str, object] | None = None

# MediaMTX (ingest) control API — same probe server-control.ps1 uses for health.
_INGEST_PROBE_URL = "http://127.0.0.1:9997/v3/config/global/get"

_stop = threading.Event()
_thread: threading.Thread | None = None


def _probe(client: httpx.Client, url: str) -> bool:
    """True if `url` answers below 500 within a short timeout. Never raises."""
    try:
        resp = client.get(url, timeout=3.0)
        return resp.status_code < 500
    except Exception:  # noqa: BLE001 — any failure = down
        return False


def _telemetry(client: httpx.Client) -> dict[str, object]:
    settings = get_settings()
    mgr = get_manager()
    statuses = mgr.status()
    running = [s for s in statuses if s.running]
    fps = sum(s.fps_inference for s in running)
    from sentry_ai import __version__

    # Per-dependency health, probed locally so superadmin can show it without
    # anyone RDP-ing into this box. `ai` is implicitly up (we're sending this).
    health = {
        "ai": True,
        "ollama": _probe(client, settings.ollama_base_url.rstrip("/") + "/api/tags"),
        "ingest": _probe(client, _INGEST_PROBE_URL),
    }

    # Resource load (CPU/RAM/GPU) for the superadmin observability dashboard
    # (docs/19). Best-effort; GPU fields are None on a CPU-only box.
    from sentry_ai import system_metrics

    resources = system_metrics.sample().as_dict()

    return {
        "fps_inference": round(fps, 1),
        "active_cameras": len(running),
        "version": __version__,
        "health": health,
        **resources,
    }


def _beat(client: httpx.Client) -> None:
    global current_config
    settings = get_settings()
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/ai-nodes/heartbeat"
    headers = {"Authorization": f"Bearer {settings.sentry_backend_service_token}"}
    resp = client.post(url, json=_telemetry(client), headers=headers, timeout=15.0)
    if resp.status_code == 200:
        current_config = resp.json()
    elif resp.status_code == 401:
        # Node was revoked (or token invalid) — stop trying noisily.
        log.warning("heartbeat.unauthorized", detail="node revoked or token invalid")
    else:
        log.warning("heartbeat.http_error", status=resp.status_code)


def _run() -> None:
    settings = get_settings()
    interval = max(15, settings.heartbeat_interval_sec)
    log.info("heartbeat.started", node_id=settings.ai_node_id, interval=interval)
    with httpx.Client() as client:
        while not _stop.is_set():
            try:
                _beat(client)
            except Exception:  # noqa: BLE001 — heartbeat must never crash the thread
                log.warning("heartbeat.failed", exc_info=True)
            _stop.wait(interval)


def start_heartbeat() -> None:
    """Start the heartbeat daemon thread (no-op if not paired or already running)."""
    global _thread
    settings = get_settings()
    if not settings.ai_node_id:
        log.info("heartbeat.disabled", reason="not paired (no ai_node_id)")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_run, name="ai-node-heartbeat", daemon=True)
    _thread.start()


def stop_heartbeat() -> None:
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=2.0)
