"""AI-node heartbeat — reports telemetry to the backend + polls config.

Runs only when this node has been paired (settings.ai_node_id is set). Posts
to /api/v1/ai-nodes/heartbeat every `heartbeat_interval_sec` using the node's
ai_node JWT (settings.sentry_backend_service_token). Best-effort: failures are
logged, never fatal. The response carries the central config (enabled,
provider, frame_skip) which we expose for new workers to read.

Runs as its OWN child PROCESS (`python -m sentry_ai.heartbeat_cli`), spawned by
the app lifespan. An in-process heartbeat (asyncio task OR daemon thread) gets
starved/killed by the GPU live-workers — the worker threads hold the GIL and
their torch CUDA context appears to wedge the heartbeat thread's NVML calls, so
the node flapped to "offline" even while healthy. A separate process has no
workers, no torch, no CUDA — it just beats, and reads the worker count over HTTP
from the running app. Bulletproof.
"""

from __future__ import annotations

import subprocess
import sys

import httpx

from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.heartbeat")

# Local app URL — the heartbeat process reads live-worker stats from here.
_LOCAL_APP = "http://127.0.0.1:8001"

# Latest config pushed from the backend (read by the manager when starting
# workers). None until the first successful heartbeat.
current_config: dict[str, object] | None = None

# MediaMTX (ingest) control API — same probe server-control.ps1 uses for health.
_INGEST_PROBE_URL = "http://127.0.0.1:9997/v3/config/global/get"

_proc: subprocess.Popen[bytes] | None = None


def _probe(client: httpx.Client, url: str) -> bool:
    """True if `url` answers below 500 within a short timeout. Never raises."""
    try:
        resp = client.get(url, timeout=3.0)
        return resp.status_code < 500
    except Exception:  # noqa: BLE001 — any failure = down
        return False


def _worker_stats(client: httpx.Client) -> tuple[float, int]:
    """(sum_fps, active_cameras) read from the running app over HTTP, since the
    heartbeat process has no in-process live-worker manager of its own."""
    try:
        r = client.get(_LOCAL_APP + "/v1/live/status", timeout=3.0)
        workers = [w for w in r.json().get("workers", []) if w.get("running")]
        fps = sum(float(w.get("fps_inference") or 0.0) for w in workers)
        return round(fps, 1), len(workers)
    except Exception:  # noqa: BLE001 — app momentarily unreachable → report 0
        return 0.0, 0


def _telemetry(client: httpx.Client) -> dict[str, object]:
    settings = get_settings()
    fps, active = _worker_stats(client)
    from sentry_ai import __version__

    # Per-dependency health, probed locally so superadmin can show it without
    # anyone RDP-ing into this box. `ai` is the app being up (we probe it).
    health = {
        "ai": _probe(client, _LOCAL_APP + "/healthz"),
        "ollama": _probe(client, settings.ollama_base_url.rstrip("/") + "/api/tags"),
        "ingest": _probe(client, _INGEST_PROBE_URL),
    }

    # Resource load (CPU/RAM/GPU) for the superadmin observability dashboard
    # (docs/19). Best-effort; GPU fields are None on a CPU-only box.
    from sentry_ai import system_metrics

    resources = system_metrics.sample().as_dict()

    return {
        "fps_inference": fps,
        "active_cameras": active,
        "version": __version__,
        "health": health,
        **resources,
    }


def _beat(client: httpx.Client) -> None:
    settings = get_settings()
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/ai-nodes/heartbeat"
    headers = {"Authorization": f"Bearer {settings.sentry_backend_service_token}"}
    resp = client.post(url, json=_telemetry(client), headers=headers, timeout=15.0)
    if resp.status_code == 401:
        log.warning("heartbeat.unauthorized", detail="node revoked or token invalid")
    elif resp.status_code != 200:
        log.warning("heartbeat.http_error", status=resp.status_code)


def run_forever() -> None:
    """Heartbeat loop — the entrypoint of the `sentry_ai.heartbeat_cli` process."""
    import time

    settings = get_settings()
    if not settings.ai_node_id:
        log.info("heartbeat.disabled", reason="not paired (no ai_node_id)")
        return
    interval = max(15, settings.heartbeat_interval_sec)
    log.info("heartbeat.started", node_id=settings.ai_node_id, interval=interval, mode="process")
    with httpx.Client() as client:
        while True:
            try:
                _beat(client)
            except Exception:  # noqa: BLE001 — heartbeat must never crash
                log.warning("heartbeat.failed", exc_info=True)
            time.sleep(interval)


def start_heartbeat() -> None:
    """Spawn the heartbeat as a separate child process (immune to the GPU
    workers starving/wedging an in-process thread). No-op if not paired."""
    global _proc
    settings = get_settings()
    if not settings.ai_node_id:
        log.info("heartbeat.disabled", reason="not paired (no ai_node_id)")
        return
    if _proc is not None and _proc.poll() is None:
        return
    _proc = subprocess.Popen([sys.executable, "-m", "sentry_ai.heartbeat_cli"])  # noqa: S603
    log.info("heartbeat.process_spawned", pid=_proc.pid)


def stop_heartbeat() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    _proc = None
