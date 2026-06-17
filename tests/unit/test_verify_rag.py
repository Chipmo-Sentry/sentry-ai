"""T08 — /v1/verify actually runs RAG retrieval when `rag_query` is set.

The backend now populates `rag_query` on both verify paths; this pins the
node-side contract: rag_query present → retrieve_context(store_id, query) is
called and its few-shot text reaches the VLM as `store_context`; rag_query
absent → no retrieval, store_context stays None.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from sentry_ai import rag
from sentry_ai.api.v1 import verify as verify_mod
from sentry_ai.schemas.verify import VerifyRequest
from sentry_ai.schemas.vlm_output import Category, VLMOutput
from sentry_ai.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CLIP_STORAGE_ROOT", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def _patched_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Stub provider/VLM/embedding; capture retrieve_context + store_context."""
    calls: dict[str, object] = {"retrieve": None, "store_context": "unset"}

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")

    monkeypatch.setattr(
        verify_mod, "get_provider", lambda name, client: SimpleNamespace(name="fake")
    )

    async def fake_verify_clip(*, clip_path, provider, store_context=None):
        calls["store_context"] = store_context
        out = VLMOutput(actions=[Category.browsing], confidence=0.9, reasoning="зүгээр үзэж байв")
        return out, 12, 5

    monkeypatch.setattr(verify_mod, "verify_clip", fake_verify_clip)

    async def fake_retrieve(store_id, query, *, k=3):
        calls["retrieve"] = (store_id, query)
        return "өмнөх ижил тохиолдлууд..."

    monkeypatch.setattr(rag, "retrieve_context", fake_retrieve)

    async def fake_embed(text):
        return [0.1, 0.2]

    monkeypatch.setattr(rag, "embed_text", fake_embed)
    return clip, calls


async def test_verify_with_rag_query_calls_retrieve(_patched_pipeline) -> None:
    clip, calls = _patched_pipeline
    store_id = uuid4()
    body = VerifyRequest(
        clip_path=str(clip),
        store_id=store_id,
        rag_query="хүн бараа авч халаасандаа нуусан байж болзошгүй",
    )
    resp = await verify_mod.verify(body, ollama=SimpleNamespace(client=None))
    # retrieve_context got the tenant filter + the backend's query text...
    assert calls["retrieve"] == (str(store_id), body.rag_query)
    # ...and its few-shot text was fed to the VLM prompt.
    assert calls["store_context"] == "өмнөх ижил тохиолдлууд..."
    assert resp.embedding == [0.1, 0.2]


async def test_verify_without_rag_query_skips_retrieve(_patched_pipeline) -> None:
    clip, calls = _patched_pipeline
    body = VerifyRequest(clip_path=str(clip), store_id=uuid4())
    await verify_mod.verify(body, ollama=SimpleNamespace(client=None))
    assert calls["retrieve"] is None
    assert calls["store_context"] is None
