"""ADR-0029 — POST /v1/edge-clip-upload verifies uploaded clip BYTES (the edge
path: the backend forwards bytes because the GPU node and the backend are
different hosts, so /v1/verify's clip_path can't be read cross-host). Mirrors
/v1/verify but multipart-in. Pins: provider + verify_clip + RAG run on the
streamed bytes, and the temp file is cleaned up."""

from __future__ import annotations

import io
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile

from sentry_ai import rag
from sentry_ai.api.v1 import verify as verify_mod
from sentry_ai.auth import assert_service_token_configured
from sentry_ai.schemas.vlm_output import Category, VLMOutput
from sentry_ai.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLIP_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("AI_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("EDGE_CLIP_MAX_MB", raising=False)
    monkeypatch.delenv("EDGE_VERIFY_MAX_QUEUE", raising=False)
    monkeypatch.delenv("EDGE_VERIFY_MAX_CONCURRENCY", raising=False)
    get_settings.cache_clear()
    # Reset the module-level GPU gate so each test gets a semaphore bound to its
    # own event loop and a clean in-flight counter (ADR-0029 §12 / I5).
    verify_mod._edge_gate = None
    verify_mod._edge_inflight = 0
    yield
    get_settings.cache_clear()
    verify_mod._edge_gate = None
    verify_mod._edge_inflight = 0


@pytest.fixture()
def _patched(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {"retrieve": None, "store_context": "unset", "clip_existed": None}

    monkeypatch.setattr(
        verify_mod, "get_provider", lambda name, client: SimpleNamespace(name="fake")
    )

    async def fake_verify_clip(*, clip_path, provider, store_context=None):
        calls["store_context"] = store_context
        calls["clip_existed"] = clip_path.exists()  # bytes were streamed to disk
        return VLMOutput(actions=[Category.pocket_conceal], confidence=0.8, reasoning="нуув"), 30, 5

    monkeypatch.setattr(verify_mod, "verify_clip", fake_verify_clip)

    async def fake_retrieve(store_id, query, *, k=3):
        calls["retrieve"] = (store_id, query)
        return "ижил тохиолдол"

    monkeypatch.setattr(rag, "retrieve_context", fake_retrieve)

    async def fake_embed(text):
        return [0.3]

    monkeypatch.setattr(rag, "embed_text", fake_embed)
    return calls


def _upload(data: bytes = b"\x00\x00\x00") -> UploadFile:
    return UploadFile(filename="clip.mp4", file=io.BytesIO(data))


async def test_edge_clip_upload_verifies_and_returns_verdict(_patched) -> None:
    calls = _patched
    resp = await verify_mod.edge_clip_upload(
        ollama=SimpleNamespace(client=None),
        clip=_upload(),
        store_id=str(uuid4()),
        provider=None,
        rag_query="халаасандаа нуув",
    )
    assert resp.category == Category.pocket_conceal
    assert resp.confidence == 0.8
    assert resp.embedding == [0.3]
    assert resp.frames_used == 5
    assert calls["clip_existed"] is True
    assert calls["store_context"] == "ижил тохиолдол"


async def test_edge_clip_upload_without_rag_skips_retrieve(_patched) -> None:
    calls = _patched
    await verify_mod.edge_clip_upload(
        ollama=SimpleNamespace(client=None),
        clip=_upload(),
        store_id=None,
        provider=None,
        rag_query=None,
    )
    assert calls["retrieve"] is None
    assert calls["store_context"] is None


# --- S0 hardening (ADR-0029 §12) ---


async def test_edge_clip_upload_rejects_oversize(_patched, monkeypatch) -> None:
    """P3: bytes past edge_clip_max_mb are rejected (413), not buffered whole."""
    monkeypatch.setenv("EDGE_CLIP_MAX_MB", "1")
    get_settings.cache_clear()
    oversize = b"\x00" * (1024 * 1024 + 1024)  # 1 MB + 1 KB
    with pytest.raises(HTTPException) as ei:
        await verify_mod.edge_clip_upload(
            ollama=SimpleNamespace(client=None),
            clip=_upload(oversize),
            store_id=None,
            provider=None,
            rag_query=None,
        )
    assert ei.value.status_code == 413
    assert verify_mod._edge_inflight == 0  # counter released on the error path


async def test_edge_clip_upload_sheds_when_saturated(_patched, monkeypatch) -> None:
    """I5: once running+waiting hits concurrency+max_queue, shed with 503."""
    monkeypatch.setenv("EDGE_VERIFY_MAX_QUEUE", "0")
    get_settings.cache_clear()
    verify_mod._edge_inflight = 1  # == concurrency(1) + queue(0) → saturated
    with pytest.raises(HTTPException) as ei:
        await verify_mod.edge_clip_upload(
            ollama=SimpleNamespace(client=None),
            clip=_upload(),
            store_id=None,
            provider=None,
            rag_query=None,
        )
    assert ei.value.status_code == 503


def test_service_token_required_in_production() -> None:
    """I6: a public verdict node must not start auth-open in staging/production."""
    with pytest.raises(RuntimeError):
        assert_service_token_configured(
            SimpleNamespace(environment="production", ai_service_token=None)
        )
    with pytest.raises(RuntimeError):
        assert_service_token_configured(SimpleNamespace(environment="staging", ai_service_token=""))
    # dev stays open; production WITH a token is allowed.
    assert (
        assert_service_token_configured(SimpleNamespace(environment="dev", ai_service_token=None))
        is None
    )
    assert (
        assert_service_token_configured(
            SimpleNamespace(environment="production", ai_service_token="tok")
        )
        is None
    )
