"""Pydantic models for live worker metadata + control APIs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskColor = Literal["green", "yellow", "red"]


class TrackPayload(BaseModel):
    """One detected person in one frame.

    L2: only person_id + box + det_confidence are populated.
    L4 (behavior scoring) fills risk_pct + color.
    """

    person_id: int = Field(description="Stable ID across frames from ByteTrack")
    box: tuple[float, float, float, float] = Field(
        description="(x1, y1, x2, y2) in source frame pixels",
    )
    det_confidence: float = Field(ge=0.0, le=1.0, description="YOLO detection score")
    risk_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    color: RiskColor = Field(default="green")


class FrameMetadata(BaseModel):
    """Per-analyzed-frame payload published to backend WS fanout."""

    camera_id: str
    frame_id: int
    ts_ms: int = Field(description="Unix ms when frame was captured")
    width: int
    height: int
    fps_inference: float = Field(description="Effective inference FPS (rolling)")
    tracks: list[TrackPayload] = Field(default_factory=list)


# === Control API schemas ===


class LiveStartRequest(BaseModel):
    camera_id: str = Field(min_length=1, max_length=64)
    rtsp_url: str = Field(min_length=1, description="Source URL — usually MediaMTX")


class LiveWorkerStatus(BaseModel):
    camera_id: str
    rtsp_url: str
    running: bool
    fps_capture: float = Field(description="Frames received per sec, rolling 5s")
    fps_inference: float = Field(description="YOLO+tracker FPS after frame_skip")
    frames_total: int
    detections_total: int
    last_error: str | None = None


class LiveStatusResponse(BaseModel):
    workers: list[LiveWorkerStatus]
