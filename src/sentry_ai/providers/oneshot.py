"""One-shot single-image VLM question, SYNC — for background classifier threads.

The main verify path (`pipeline/verifier.py`) is async and speaks the strict
theft-category `VLMOutput` schema. Attribute questions ("is this person wearing
a staff badge?") don't fit that schema and are asked from plain worker threads
(e.g. `live_worker/staff.py`'s verifier thread), so this helper is deliberately
synchronous httpx with a free-form JSON reply.

Transport follows the EFFECTIVE provider (central control, ADR-0026): the vLLM
OpenAI endpoint when a vllm provider is selected, else Ollama /api/chat with the
provider's own model tag. Failures return None — attribute classification is
best-effort and must never break the caller.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from sentry_ai.logging_setup import get_logger
from sentry_ai.providers.factory import get_provider_class, resolve_provider_name
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.providers.oneshot")


def ask_json(prompt: str, image_jpeg: bytes, timeout_sec: float = 20.0) -> dict[str, Any] | None:
    """Ask the effective VLM one JSON question about one JPEG. None on any failure."""
    try:
        name = resolve_provider_name(None)
        cls = get_provider_class(name)
        runtime = getattr(cls, "runtime", "ollama") if cls is not None else "ollama"
        if runtime == "vllm":
            raw = _ask_vllm(prompt, image_jpeg, timeout_sec)
        else:
            raw = _ask_ollama(prompt, image_jpeg, timeout_sec, cls)
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception as e:  # noqa: BLE001 — best-effort by contract
        log.warning("oneshot.ask_failed", error=str(e))
        return None


def _ask_vllm(prompt: str, image_jpeg: bytes, timeout_sec: float) -> str:
    s = get_settings()
    b64 = base64.b64encode(image_jpeg).decode("ascii")
    payload: dict[str, Any] = {
        "model": s.vllm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "stream": False,
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    with httpx.Client(base_url=s.vllm_base_url.rstrip("/"), timeout=timeout_sec) as client:
        resp = client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError(f"Unexpected vLLM reply shape: {data!r}")
    return content


def _ask_ollama(prompt: str, image_jpeg: bytes, timeout_sec: float, cls: type | None) -> str:
    s = get_settings()
    model = getattr(cls, "model_tag", None) or "qwen3-vl:4b-instruct"
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image_jpeg).decode("ascii")],
            }
        ],
    }
    with httpx.Client(base_url=s.ollama_base_url.rstrip("/"), timeout=timeout_sec) as client:
        resp = client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content:
        raise ValueError(f"Unexpected /api/chat reply shape: {data!r}")
    return content
