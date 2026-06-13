"""Qwen3-VL-4B via a vLLM OpenAI-compatible server (scale-path runtime, ADR-0026).

Ollama is the default on the laptop node (small shared GPU, bursty single-request
workload). vLLM is the runtime for the future dedicated Linux GPU VPS, where
batched serving across many cameras/stores pays off. This provider speaks the
OpenAI `/v1/chat/completions` API that vLLM exposes (`vllm serve` >= 0.11.0):
images go as `image_url` data URIs, output is constrained with
`response_format={"type": "json_object"}`.

VRAM/context fit is a server-side concern on the vLLM host (`--max-model-len`),
so — unlike the Ollama path — no num_ctx is sent per request.
"""

import base64
import json
from typing import Any

import httpx
from pydantic import ValidationError

from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.schemas.vlm_output import VLMOutput, VLMParseError
from sentry_ai.settings import get_settings


class Qwen3VLvLLMProvider:
    """Implements `VLMProvider`. Talks to a vLLM OpenAI-compatible endpoint."""

    name = "qwen3-vl-vllm"
    runtime = "vllm"

    def __init__(self, ollama: OllamaClient) -> None:
        # The factory hands every provider the shared OllamaClient; vLLM doesn't use
        # it (different server + API) — it reads its endpoint/model from settings.
        del ollama
        s = get_settings()
        self._base_url = s.vllm_base_url.rstrip("/")
        self._model = s.vllm_model
        self._timeout = s.inference_timeout_sec

    async def verify(
        self,
        frames: list[bytes],
        prompt: str,
        timeout_sec: int,
    ) -> VLMOutput:
        del timeout_sec  # uses the configured timeout
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in frames:
            b64 = base64.b64encode(img).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        raw = ""
        try:
            raw = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            return VLMOutput.model_validate(parsed)
        except (KeyError, IndexError, TypeError) as e:
            raise VLMParseError(str(data), e) from e
        except (json.JSONDecodeError, ValidationError) as e:
            raise VLMParseError(raw, e) from e
