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
    # RAG (docs/19 Phase 4): a short description of the event. When set, the node
    # retrieves similar past staff-verified cases for this store and feeds them
    # to the VLM as few-shot context. Omit to skip RAG.
    rag_query: str | None = Field(default=None, max_length=2000)


class VerifyResponse(BaseModel):
    """Returned synchronously from /v1/verify."""

    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    model_name: str
    inference_latency_ms: int = Field(ge=0)
    frames_used: int
    # Embedding of `reasoning` (docs/19 Phase 4 RAG). The backend stores it on
    # the alert so staff feedback can spawn a verified_case. None if RAG is off.
    embedding: list[float] | None = None
