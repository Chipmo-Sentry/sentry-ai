"""/v1/live router — control + status endpoints for live worker."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Response

from sentry_ai.auth import require_service_token
from sentry_ai.live_worker import get_manager
from sentry_ai.live_worker.schemas import (
    LiveStartRequest,
    LiveStatusResponse,
)
from sentry_ai.runtime import is_railway_host
from sentry_ai.settings import get_settings

router = APIRouter(prefix="/v1/live", tags=["live"], dependencies=[Depends(require_service_token)])


def _validate_rtsp_url(url: str) -> None:
    """Reject non-allowlisted URL schemes — blocks SSRF / local-file vectors
    (file://, http://, etc.) before the worker opens the URL with OpenCV."""
    allowed = {
        s.strip().lower() for s in get_settings().allowed_rtsp_schemes.split(",") if s.strip()
    }
    scheme = urlparse(url).scheme.lower()
    if scheme not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"rtsp_url scheme '{scheme}' not allowed (allowed: {sorted(allowed)})",
        )


@router.post("/start", status_code=202)
def start(req: LiveStartRequest) -> dict[str, str]:
    # Red-line #2 / ADR-0007: RTSP ingest must NEVER run on Railway.
    if is_railway_host():
        raise HTTPException(
            status_code=403,
            detail="Live RTSP workers must not run on Railway — deploy sentry-ai on the GPU/VPS host",
        )
    _validate_rtsp_url(req.rtsp_url)
    get_manager().start_camera(req.camera_id, req.rtsp_url, store_id=req.store_id)
    return {"camera_id": req.camera_id, "status": "starting"}


@router.post("/stop/{camera_id}")
def stop(camera_id: str) -> dict[str, str]:
    stopped = get_manager().stop_camera(camera_id)
    return {"camera_id": camera_id, "status": "stopped" if stopped else "not_found"}


@router.get("/status", response_model=LiveStatusResponse)
def status() -> LiveStatusResponse:
    return LiveStatusResponse(workers=get_manager().status())


@router.get("/emitter")
def emitter_stats() -> dict[str, int]:
    return get_manager().emitter_stats


@router.get("/provider")
def provider_status() -> dict[str, object]:
    """Effective VLM provider this node will actually use + readiness, so the
    heartbeat (and thus the dashboard) can show whether a central provider change
    was really applied on the server. Set by the config poller."""
    from sentry_ai.runtime_config import get_provider_health

    h = get_provider_health()
    return {"effective": h.effective, "ready": h.ready, "error": h.error}


@router.get(
    "/snapshot/{camera_id}",
    responses={200: {"content": {"image/jpeg": {}}}, 404: {}},
)
def snapshot(camera_id: str) -> Response:
    """Return latest annotated frame as JPEG — debug viewer.

    Open in browser. Browser caches aggressively — append `?t=<unix>` to refresh.
    """
    result = get_manager().snapshot(camera_id)
    if result is None:
        raise HTTPException(status_code=404, detail="camera_id not running or no frame yet")
    jpeg_bytes, ts = result
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Snapshot-Timestamp": str(ts),
        },
    )
