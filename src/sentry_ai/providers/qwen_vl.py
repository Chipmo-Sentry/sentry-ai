"""Qwen2.5-VL-7B provider — Alibaba's 7B parameter VLM, strong at
structured JSON output and multilingual reasoning."""

import json

from pydantic import ValidationError

from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.schemas.vlm_output import VLMOutput, VLMParseError


class QwenVLProvider:
    """Implements `VLMProvider`. Tag in Ollama: `qwen2.5vl:7b`."""

    name = "qwen2.5-vl-7b"
    _model_tag = "qwen2.5vl:7b"

    def __init__(self, ollama: OllamaClient):
        self._ollama = ollama

    async def verify(
        self,
        frames: list[bytes],
        prompt: str,
        timeout_sec: int,
    ) -> VLMOutput:
        del timeout_sec  # uses OllamaClient's configured timeout
        raw = await self._ollama.chat_with_images(
            model=self._model_tag,
            prompt=prompt,
            images=frames,
            format_json=True,
        )
        try:
            parsed = json.loads(raw)
            return VLMOutput.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            raise VLMParseError(raw, e) from e
