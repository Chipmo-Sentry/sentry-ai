"""Application settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["dev", "staging", "production"] = "dev"
    debug: bool = False

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8001  # different from backend so they can coexist on localhost

    # Ollama (host runs the runtime; we just HTTP into it)
    ollama_base_url: str = "http://localhost:11434"
    default_provider: str = "minicpm-v-2.6"
    # RAG feedback loop (docs/19 Phase 4): Ollama embedding model for similarity.
    # Empty disables RAG. `ollama pull nomic-embed-text` on the node first.
    embed_model: str = "nomic-embed-text"
    inference_timeout_sec: int = 30
    retry_on_parse_error: int = 2

    # Frame extraction (Stage 2 input)
    frames_per_clip: int = 5
    frame_max_dim: int = 640
    frame_jpeg_quality: int = 85

    # Backend integration (sentry-ai → sentry-backend)
    sentry_backend_url: str = "http://localhost:8000"
    # Bearer token sent to the backend. After pairing this holds the node's
    # ai_node JWT (the backend accepts it for live-metadata too).
    sentry_backend_service_token: str = "dev-service-token"

    # Set by `python -m sentry_ai.pair` after a successful pairing. When present,
    # the node runs a heartbeat loop reporting telemetry + polling config.
    ai_node_id: str | None = None
    heartbeat_interval_sec: int = 60

    # M1-LIVE L2: live worker auto-start
    # Comma-separated list of `camera_id=rtsp_url` pairs. Empty = no auto-start.
    # Example:
    #   live_auto_start="cam1_hik=rtsp://localhost:8554/cam1_hik,cam2_unv=rtsp://localhost:8554/cam2_unv"
    live_auto_start: str = ""
    # Store id applied to ALL auto-started cameras so they share one cross-camera
    # re-ID registry (ADR-0023) → the same person gets the same store_person_id
    # across the store's cameras. None = per-camera ids only.
    live_auto_start_store_id: str | None = None
    live_frame_skip: int = 3  # analyze every Nth frame (10 FPS on 30 FPS source)

    # YOLO weights (#3, ADR-0026). Pose drives ALL behavior detection, so accuracy
    # here is foundational — YOLO26 is NMS-free end-to-end and its RLE head scores
    # occluded joints better than YOLO11 (+~7 AP COCO-pose). Ultralytics
    # auto-downloads on first run (needs ultralytics>=8.4). Override to
    # yolo26n-pose.pt for low-power CPU hosts, yolo26m-pose.pt for max accuracy,
    # or yolo11s-pose.pt/yolo11n.pt to roll back to the pre-ADR-0026 stack.
    yolo_pose_weights: str = "yolo26s-pose.pt"
    yolo_item_weights: str = "yolo26n.pt"

    # Cross-camera re-ID embedder (#4). "histogram" = dependency-light color
    # histogram (default, weak). "osnet" = learned torchreid OSNet (robust, needs
    # the optional torchreid+torch deps + ideally a GPU; falls back to histogram
    # if unavailable). See live_worker/reid.py::make_embedder.
    reid_model: str = "histogram"

    # --- Service auth + input hardening (enforce-if-configured) ---
    # Shared secret required as `Authorization: Bearer <token>` on /v1/* routes.
    # None/empty → routes are OPEN (trusted-LAN M1 default). Set the SAME value
    # as sentry-backend's SENTRY_AI_SERVICE_TOKEN before exposing this service.
    ai_service_token: str | None = None
    # When set, /v1/verify clip_path must resolve INSIDE this directory — blocks
    # the arbitrary-host-file read. None → no constraint (LAN default).
    clip_storage_root: str | None = None
    # MediaMTX recordings dir on THIS node — /v1/cut-verify cuts the breach clip
    # from here (docs/19 I5). Relative to the sentry-ingest box's recordings/.
    mediamtx_recordings_dir: str | None = None
    # Schemes /v1/live/start may open with cv2.VideoCapture — blocks file://,
    # http(s):// and other SSRF/local-file vectors. rtsp(s) only by default.
    allowed_rtsp_schemes: str = "rtsp,rtsps"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
