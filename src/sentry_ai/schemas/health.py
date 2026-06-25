"""Health-check schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_reachable: bool
    # Whether the effective VLM provider actually needs Ollama. False on a vLLM node,
    # where Ollama being unreachable does NOT degrade verify (avoids a false alarm).
    ollama_required: bool = True
    loaded_models: list[str]
