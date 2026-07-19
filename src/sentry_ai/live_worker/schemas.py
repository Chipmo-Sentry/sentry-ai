"""Pydantic models for live worker metadata + control APIs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskColor = Literal["green", "yellow", "red"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TrackPayload(BaseModel):
    """One detected person in one frame.

    L2: only person_id + box + det_confidence are populated.
    L4 (behavior scoring) fills risk_pct + color.
    """

    person_id: int = Field(description="Per-camera ByteTrack id (resets per camera)")
    box: tuple[float, float, float, float] = Field(
        description="(x1, y1, x2, y2) in source frame pixels",
    )
    det_confidence: float = Field(ge=0.0, le=1.0, description="YOLO detection score")
    # Absolute 0-100 risk for THIS camera's track (ADR-0024).
    risk_pct: float = Field(default=0.0, ge=0.0)
    color: RiskColor = Field(default="green")
    # v2 behavior engine (ADR-0024): 4-level band, per-track state-machine state,
    # and the sequence rules that fired this episode.
    level: RiskLevel = Field(default="LOW")
    state: str = Field(default="IDLE", description="Behavior state machine state")
    sequences: list[str] = Field(default_factory=list, description="Triggered sequence rule keys")
    # Episode-accumulated criteria (stable keys, first-fired order) + per-key
    # banked score, and the Mongolian display strings for direct overlay use.
    behaviors: list[str] = Field(
        default_factory=list, description="Fired criteria keys this episode, first-fired order"
    )
    behavior_scores: dict[str, float] = Field(
        default_factory=dict, description="Accumulated score contribution per criterion key"
    )
    behavior_offsets: dict[str, float] = Field(
        default_factory=dict,
        description="Seconds from episode start to each criterion's first firing",
    )
    reasons: list[str] = Field(
        default_factory=list, description="Mongolian display strings (latest non-empty frame)"
    )
    episode_started_ms: int | None = Field(
        default=None, description="Unix ms when this episode's first criterion fired"
    )
    # Cross-camera re-ID (ADR-0022/0023): store-global person id + their risk
    # accumulated across the store's cameras. None when re-ID is disabled.
    store_person_id: int | None = Field(default=None)
    store_risk_pct: float | None = Field(default=None)
    # docs/30 F5 — visitor demographics from the node's CPU classifier
    # (live_worker/demographics.py). None until the per-track vote is stable:
    # the backend counts each (camera, person_id) once, FIRST values win, so a
    # premature label would permanently mis-bucket the visitor.
    gender: Literal["male", "female", "unknown"] | None = Field(default=None)
    age_band: Literal["child", "youth", "adult", "senior", "unknown"] | None = Field(default=None)
    # Staff identification via neck-badge color (live_worker/staff.py). True only
    # after the per-track vote locked (color evidence, VLM-confirmed when
    # available). Riding every frame so the overlay can recolor the box AND the
    # backend aggregator can exclude staff from visitor KPIs/paths/heatmap.
    # Alerts are NOT suppressed for staff (internal theft stays monitored).
    is_staff: bool = Field(default=False)


class ItemPayload(BaseModel):
    """One detected item (merchandise) in one frame, for the live overlay.

    `held` marks an item a person's wrist is over (the 'in hand' geometry) so the
    browser can highlight + name what's being carried. `label_mn` is the Mongolian
    display name computed node-side so the frontend just renders text."""

    box: tuple[float, float, float, float] = Field(
        description="(x1, y1, x2, y2) in source frame pixels",
    )
    label: str = Field(description="Detector class label (English)")
    label_mn: str = Field(description="Mongolian display name (falls back to label)")
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    held: bool = Field(default=False, description="A wrist is within reach of this item box")


class FrameMetadata(BaseModel):
    """Per-analyzed-frame payload published to backend WS fanout."""

    camera_id: str
    frame_id: int
    ts_ms: int = Field(description="Unix ms when frame was captured")
    width: int
    height: int
    fps_inference: float = Field(description="Effective inference FPS (rolling)")
    tracks: list[TrackPayload] = Field(default_factory=list)
    # Detected merchandise this frame (for the live overlay's held-item box+name).
    items: list[ItemPayload] = Field(default_factory=list)


# === Control API schemas ===


class Zone(BaseModel):
    """A per-camera detection polygon in normalized 0-1 image space (docs/29).

    Lenient: the backend already validates on write, so the node accepts what it's
    given (extra fields ignored) and the worker's compile step drops anything
    malformed rather than rejecting the whole start request."""

    type: str
    points: list[tuple[float, float]] = Field(default_factory=list)
    id: str | None = None


class LiveStartRequest(BaseModel):
    camera_id: str = Field(min_length=1, max_length=64)
    rtsp_url: str = Field(min_length=1, description="Source URL — usually MediaMTX")
    store_id: str | None = Field(
        default=None, description="Enables store-scoped cross-camera re-ID (ADR-0023)"
    )
    risk_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Per-camera breach threshold (0-100). None → node-global default.",
    )
    # docs/29 P1c — per-camera detection zones; drive the zone-aware criteria
    # (exit_after_concealment / repeated_shelf_visit). None/empty → no zone scoring.
    zones: list[Zone] | None = Field(default=None)


class LiveWorkerStatus(BaseModel):
    camera_id: str
    rtsp_url: str
    running: bool
    fps_capture: float = Field(description="Frames received per sec, rolling 5s")
    fps_inference: float = Field(description="YOLO+tracker FPS after frame_skip")
    frames_total: int
    detections_total: int
    # Effective (live) frame_skip applied on the latest read + persons in the
    # latest analyzed frame — surfaced so the heartbeat reports real per-camera
    # YOLO numbers and so a superadmin frame_skip edit is visibly applied.
    frame_skip: int = 3
    persons: int = 0
    last_error: str | None = None


class LiveStatusResponse(BaseModel):
    workers: list[LiveWorkerStatus]
