"""AI-node heartbeat — reports telemetry to the backend + polls config.

Runs only when this node has been paired (settings.ai_node_id is set). Posts
to /api/v1/ai-nodes/heartbeat every `heartbeat_interval_sec` using the node's
ai_node JWT (settings.sentry_backend_service_token). Best-effort: failures are
logged, never fatal. The response carries the central config (enabled,
provider, frame_skip) which we expose for new workers to read.
"""

from __future__ import annotations

import asyncio

import httpx

from sentry_ai.live_worker import get_manager
from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.heartbeat")

# Latest config pushed from the backend (read by the manager when starting
# workers). None until the first successful heartbeat.
current_config: dict[str, object] | None = None


def _telemetry() -> dict[str, object]:
    mgr = get_manager()
    statuses = mgr.status()
    running = [s for s in statuses if s.running]
    fps = sum(s.fps_inference for s in running)
    from sentry_ai import __version__

    return {
        "fps_inference": round(fps, 1),
        "active_cameras": len(running),
        "version": __version__,
    }


async def _beat(client: httpx.AsyncClient) -> None:
    global current_config
    settings = get_settings()
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/ai-nodes/heartbeat"
    headers = {"Authorization": f"Bearer {settings.sentry_backend_service_token}"}
    resp = await client.post(url, json=_telemetry(), headers=headers, timeout=15.0)
    if resp.status_code == 200:
        current_config = resp.json()
    elif resp.status_code == 401:
        # Node was revoked (or token invalid) — stop trying noisily.
        log.warning("heartbeat.unauthorized", detail="node revoked or token invalid")
    else:
        log.warning("heartbeat.http_error", status=resp.status_code)


async def heartbeat_loop() -> None:
    settings = get_settings()
    if not settings.ai_node_id:
        log.info("heartbeat.disabled", reason="not paired (no ai_node_id)")
        return
    interval = max(15, settings.heartbeat_interval_sec)
    log.info("heartbeat.started", node_id=settings.ai_node_id, interval=interval)
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await _beat(client)
            except Exception:  # noqa: BLE001 — heartbeat must never crash the app
                log.warning("heartbeat.failed", exc_info=True)
            await asyncio.sleep(interval)
