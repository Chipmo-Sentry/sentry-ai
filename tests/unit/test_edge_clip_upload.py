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
from fastapi import UploadFile

from sentry_ai import rag
from sentry_ai.api.v1 import verify as verify_mod
from sentry_ai.schemas.vlm_output import Category, VLMOutput
from sentry_ai.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLIP_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("AI_SERVICE_TOKEN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
