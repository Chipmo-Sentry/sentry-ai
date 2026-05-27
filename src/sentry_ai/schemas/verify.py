"""POST /v1/verify request + response schemas."""

from uuid import UUID

from pydantic import BaseModel, Field

from sentry_ai.schemas.vlm_output import Category


class VerifyRequest(BaseModel):
    """Sent by sentry-backend after a clip arrives."""

    clip_path: str = Field(min_length=1, description="Absolute path on the host filesystem")
    store_id: UUID | None = None
    camera_id: UUID | None = None
    provider: str | None = Field(default=None, description="Override default_provider (A/B test)")


class VerifyResponse(BaseModel):
    """Returned synchronously from /v1/verify."""

    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    model_name: str
    inference_latency_ms: int = Field(ge=0)
    frames_used: int
