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


class CutVerifyRequest(BaseModel):
    """Live-breach verify (docs/19 I5): the backend asks this node to cut a clip
    from its own MediaMTX recordings and verify it. Sent on a threshold breach."""

    # Safe slug only — flows into a filesystem path under the recordings dir,
    # so reject '/', '..', whitespace etc. (matches backend MEDIAMTX_PATH_PATTERN).
    mediamtx_path: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    store_id: UUID | None = None
    camera_id: UUID | None = None
    person_id: int | None = None
    peak_risk_pct: float | None = None
    start_offset_sec: int = -5
    # Cap covers the dynamic episode window (backend caps at live_clip_max_sec,
    # default 90): pre-roll back to the episode's first criterion + post-roll.
    duration_sec: int = Field(default=15, ge=1, le=120)
    provider: str | None = None
    rag_query: str | None = Field(default=None, max_length=2000)


class CutVerifyResponse(VerifyResponse):
    """VerifyResponse + the cut clip (base64 mp4) so the backend can store it."""

    duration_sec_clip: float
    captured_at: str  # ISO-8601 UTC
    file_size_bytes: int
    clip_b64: str  # base64-encoded mp4
