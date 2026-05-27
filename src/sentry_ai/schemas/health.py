"""Health-check schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_reachable: bool
    loaded_models: list[str]
