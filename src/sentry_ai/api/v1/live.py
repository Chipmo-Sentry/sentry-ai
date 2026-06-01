"""/v1/live router — control + status endpoints for live worker."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from sentry_ai.live_worker import get_manager
from sentry_ai.live_worker.schemas import (
    LiveStartRequest,
    LiveStatusResponse,
)

router = APIRouter(prefix="/v1/live", tags=["live"])


@router.post("/start", status_code=202)
def start(req: LiveStartRequest) -> dict[str, str]:
    get_manager().start_camera(req.camera_id, req.rtsp_url)
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
