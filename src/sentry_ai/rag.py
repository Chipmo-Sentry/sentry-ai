"""RAG feedback loop (docs/19 Phase 4) — the AI-node side.

Two best-effort operations, both no-ops if `embed_model` is unset or anything
fails (RAG must never break a verification):

  • retrieve_context(store_id, query) — embed the query (Ollama), ask the backend
    for the most similar past staff-verified cases, and format them as few-shot
    text to drop into the VLM verify prompt (`store_context`). So the model sees
    "here's how staff judged similar events at this store" and drifts toward the
    store's real patterns — without retraining.
  • store_case(store_id, verdict, category, description) — embed a confirmed
    event and persist it for future retrieval.

The model weights never change; the *context* improves. Embeddings come from
Ollama (`nomic-embed-text` by default), similarity ranking is done server-side.
"""

from __future__ import annotations

import httpx

from sentry_ai.logging_setup import get_logger
from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.rag")


async def _embed(text: str) -> list[float] | None:
    settings = get_settings()
    if not settings.embed_model:
        return None
    client = OllamaClient(settings.ollama_base_url, timeout_sec=30)
    try:
        return await client.embed(settings.embed_model, text)
    except Exception as e:  # noqa: BLE001 — RAG is best-effort
        log.info("rag.embed_failed", error=str(e))
        return None
    finally:
        await client.aclose()


def _backend_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_settings().sentry_backend_service_token}"}


async def retrieve_context(
    store_id: str | None, query: str, *, k: int = 3
) -> str | None:
    """Few-shot text from the top-k similar past verified cases, or None."""
    embedding = await _embed(query)
    if embedding is None:
        return None
    settings = get_settings()
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/internal/rag/similar"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json={"store_id": store_id, "embedding": embedding, "k": k},
                headers=_backend_headers(),
            )
        if resp.status_code != 200:
            log.info("rag.retrieve_non_200", status=resp.status_code)
            return None
        matches = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        log.info("rag.retrieve_failed", error=str(e))
        return None
    return _format(matches)


def _format(matches: list[dict[str, object]]) -> str | None:
    if not matches:
        return None
    verdict_label = {
        "true_positive": "БАТЛАГДСАН хулгай",
        "false_positive": "ХУДАЛ сэрэлт (хулгай биш)",
        "unclear": "тодорхойгүй",
    }
    lines = [
        "Энэ дэлгүүрийн өмнөх ижил төстэй тохиолдлуудыг ажилтан дараах байдлаар үнэлсэн:"
    ]
    for m in matches:
        v = verdict_label.get(str(m.get("verdict")), str(m.get("verdict")))
        desc = str(m.get("description", "")).strip()
        lines.append(f"- «{desc}» → {v}")
    lines.append("Үүнийг лавлагаа болгон, гэхдээ одоогийн дүр зургийг бие даан дүгнэ.")
    return "\n".join(lines)


async def store_case(
    store_id: str | None,
    *,
    verdict: str,
    category: str | None,
    description: str,
) -> bool:
    """Persist a staff-verified case for future RAG retrieval. Best-effort."""
    embedding = await _embed(description)
    if embedding is None:
        return False
    settings = get_settings()
    url = settings.sentry_backend_url.rstrip("/") + "/api/v1/internal/rag/cases"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json={
                    "store_id": store_id,
                    "verdict": verdict,
                    "category": category,
                    "description": description,
                    "embedding": embedding,
                },
                headers=_backend_headers(),
            )
        return resp.status_code in (200, 201)
    except httpx.HTTPError as e:
        log.info("rag.store_failed", error=str(e))
        return False
