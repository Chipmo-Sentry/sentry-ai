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
    sentry_backend_service_token: str = "dev-service-token"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
