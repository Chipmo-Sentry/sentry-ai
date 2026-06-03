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
    live_frame_skip: int = 3  # analyze every Nth frame (10 FPS on 30 FPS source)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
