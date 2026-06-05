"""RAG feedback-loop helper (docs/19 Phase 4)."""

from __future__ import annotations

import httpx
import pytest
import respx

from sentry_ai import rag


def test_format_builds_few_shot() -> None:
    out = rag._format(
        [
            {"description": "хүн бараа халаасандаа хийв", "verdict": "true_positive"},
            {"description": "зүгээр л үзэж байв", "verdict": "false_positive"},
        ]
    )
    assert out is not None
    assert "халаасандаа" in out
    assert "БАТЛАГДСАН" in out
    assert "ХУДАЛ" in out


def test_format_empty_is_none() -> None:
    assert rag._format([]) is None


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_context_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    monkeypatch.setenv("SENTRY_BACKEND_URL", "http://backend.test")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    rag.get_settings.cache_clear()  # type: ignore[attr-defined]

    respx.post("http://ollama.test/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
    )
    similar = respx.post("http://backend.test/api/v1/internal/rag/similar").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "description": "халаасанд нуув",
                    "verdict": "true_positive",
                    "category": "pocket_conceal",
                    "score": 0.9,
                }
            ],
        )
    )

    ctx = await rag.retrieve_context("store-1", "person concealing item")
    assert ctx is not None and "халаасанд нуув" in ctx
    assert similar.called


@pytest.mark.asyncio
@respx.mock
async def test_store_case_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    monkeypatch.setenv("SENTRY_BACKEND_URL", "http://backend.test")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    rag.get_settings.cache_clear()  # type: ignore[attr-defined]

    respx.post("http://ollama.test/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [0.5, 0.5]})
    )
    cases = respx.post("http://backend.test/api/v1/internal/rag/cases").mock(
        return_value=httpx.Response(201, json={"id": "case-1"})
    )

    ok = await rag.store_case(
        "store-1", verdict="true_positive", category="pocket_conceal", description="x"
    )
    assert ok is True
    assert cases.called


@pytest.mark.asyncio
async def test_retrieve_disabled_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_MODEL", "")
    rag.get_settings.cache_clear()  # type: ignore[attr-defined]
    assert await rag.retrieve_context("store-1", "anything") is None
