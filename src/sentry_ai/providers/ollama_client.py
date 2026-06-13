"""Thin async wrapper over the Ollama HTTP API.

Ollama exposes an OpenAI-compatible chat endpoint plus its native /api/generate.
For multimodal we use /api/chat with role=user + images[] (base64).
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import base64
from typing import Any

import httpx


class OllamaClient:
    def __init__(self, base_url: str, timeout_sec: int, num_ctx: int = 0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_sec)
        # >0 → sent as options.num_ctx to cap the KV cache so the VLM fits on the
        # GPU (see settings.vlm_num_ctx). 0 → omit, use Ollama's per-model default.
        self._num_ctx = num_ctx

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_with_images(
        self,
        model: str,
        prompt: str,
        images: list[bytes],
        *,
        format_json: bool = True,
    ) -> str:
        """Send a single user turn with image attachments. Returns the raw text reply."""
        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(img).decode("ascii") for img in images],
                }
            ],
        }
        if format_json:
            # Tells Ollama to constrain the output to valid JSON
            payload["format"] = "json"
        if self._num_ctx > 0:
            # Cap the context so the KV cache stays small enough for full-GPU offload.
            payload["options"] = {"num_ctx": self._num_ctx}

        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Native Ollama shape: {"message": {"role": "assistant", "content": "..."}, ...}
        content = data.get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"Unexpected /api/chat reply shape: {data!r}")
        return content

    async def embed(self, model: str, text: str) -> list[float]:
        """Embed `text` via Ollama /api/embeddings → a float vector.

        Used by the RAG feedback loop (docs/19 Phase 4) to find similar past
        verified cases. Raises on transport/parse errors so the caller can fall
        back gracefully (RAG is best-effort).
        """
        resp = await self._client.post("/api/embeddings", json={"model": model, "prompt": text})
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise ValueError(f"Unexpected /api/embeddings shape: {data!r}")
        return [float(x) for x in vec]

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    async def is_reachable(self) -> bool:
        try:
            resp = await self._client.get("/api/tags", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
