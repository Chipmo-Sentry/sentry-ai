"""Qwen3-VL-4B provider — successor to Qwen2.5-VL (maintenance-only since 2026).
~92-95% of Qwen2.5-VL-7B accuracy at ~58% of the parameters, ~40-50% less VRAM
(ADR-0026) — the alt slot to A/B against MiniCPM-V on verified_cases."""

import json

from pydantic import ValidationError

from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.schemas.vlm_output import VLMOutput, VLMParseError


class Qwen3VLProvider:
    """Implements `VLMProvider`. Tag in Ollama: `qwen3-vl:4b-instruct`."""

    name = "qwen3-vl-4b"
    # Explicit -instruct: the bare :4b tag may resolve to the "thinking" variant,
    # whose chain-of-thought preamble breaks format_json parsing.
    _model_tag = "qwen3-vl:4b-instruct"

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
