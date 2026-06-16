"""Application settings."""

from functools import lru_cache
from pathlib import Path
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
    default_provider: str = "qwen3-vl-4b"
    # RAG feedback loop (docs/19 Phase 4): Ollama embedding model for similarity.
    # Empty disables RAG. `ollama pull nomic-embed-text` on the node first.
    embed_model: str = "nomic-embed-text"
    inference_timeout_sec: int = 30
    retry_on_parse_error: int = 2
    # vLLM provider (ADR-0026 scale-path). The `qwen3-vl-vllm` provider talks to a
    # vLLM OpenAI-compatible server (`vllm serve Qwen/Qwen3-VL-4B-Instruct`,
    # vllm>=0.11.0) — the runtime for the future dedicated Linux GPU VPS, NOT this
    # Windows laptop node (vLLM needs Linux + a dedicated GPU). Context/VRAM fit is
    # configured server-side there via --max-model-len, so no num_ctx is sent here.
    vllm_base_url: str = "http://localhost:8000/v1"  # OpenAI-compatible base (remote)
    vllm_model: str = "Qwen/Qwen3-VL-4B-Instruct"  # served model id on the vLLM host

    # VLM context window (tokens) sent as Ollama `options.num_ctx`. Ollama otherwise
    # defaults to the model's max trained context (Qwen3-VL = 262144), whose KV cache
    # + compute buffers blow past 8 GB VRAM → only a few layers offload to GPU and the
    # rest fall back to CPU (verify becomes minutes-slow). We send ~5 small frames
    # (frame_max_dim 640) + a short prompt and expect short JSON. Measured on an
    # 8 GB RTX 4060 co-resident with the live YOLO workers: 8192 → 100% GPU offload
    # (~7 s verify); 16384 → 83% GPU (spills to CPU). 0 = don't send (Ollama default).
    vlm_num_ctx: int = 8192

    # Frame extraction (Stage 2 input)
    frames_per_clip: int = 5
    frame_max_dim: int = 640
    frame_jpeg_quality: int = 85
    # ffmpeg binary for clip cutting. Bare "ffmpeg" works when it's on PATH; set
    # an absolute path when the node runs as a service without ffmpeg on PATH
    # (e.g. FFMPEG_PATH=D:/ChipmoSentryAI/bin/ffmpeg.exe).
    ffmpeg_path: str = "ffmpeg"
    # ffprobe binary (clip-duration probe in pipeline/frames.py). Optional override;
    # when unset it's derived from ffmpeg_path so setting FFMPEG_PATH alone is enough
    # (.../ffmpeg.exe → .../ffprobe.exe). ffprobe ships alongside ffmpeg.
    ffprobe_path_override: str = ""

    @property
    def ffprobe_path(self) -> str:
        """Resolved ffprobe binary: explicit override, else derived from
        ffmpeg_path (.../ffmpeg.exe → .../ffprobe.exe), else bare "ffprobe"."""
        if self.ffprobe_path_override:
            return self.ffprobe_path_override
        fp = self.ffmpeg_path
        if fp and fp != "ffmpeg" and ("/" in fp or "\\" in fp):
            p = Path(fp)
            return str(p.with_name(p.name.replace("ffmpeg", "ffprobe")))
        return "ffprobe"

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

    # --- Node-push live alerts ---
    # The node detects a sustained risk breach itself, cuts + VLM-verifies
    # LOCALLY, and POSTs the finished alert to the backend
    # (/api/v1/internal/live-alert). Outbound = reliable, unlike the retired
    # cloud→node /v1/cut-verify pull. Disable to stop pushing (backend must then
    # use its own pull path — see backend live_alerts_via_node_push).
    live_alert_push_enabled: bool = True
    # Risk % a tracked person must reach to trigger a breach (matches the
    # backend's per-camera risk_threshold default; yellow band). Sustained for
    # sustain_sec, then debounced per person for cooldown_sec.
    live_alert_threshold_pct: float = 11.0
    # VLM-primary detection (ADR pivot 2026-06-16): instead of the geometric
    # behaviour engine GATING the VLM, scan a present person with the VLM on this
    # cadence and let the AI decide. The behaviour engine now only prioritises
    # WHICH person to scan + drives the overlay. Throttled by the single GPU
    # worker (a scan in flight skips the next ticks), so this is a target, not a
    # guarantee — effective rate ≈ VLM latency when busy.
    live_scan_interval_sec: float = 3.0
    live_breach_sustain_sec: float = 1.0
    live_breach_cooldown_sec: float = 30.0
    # Wait this long after the breach before cutting, so the clip catches the act
    # COMPLETING; plus clip-window pre-pad + a hard cap on total length.
    live_breach_post_roll_sec: float = 10.0
    live_clip_pre_pad_sec: int = 3
    live_clip_max_sec: int = 45

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
