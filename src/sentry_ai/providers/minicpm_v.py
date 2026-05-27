"""MiniCPM-V 2.6 provider — 8B parameter VLM, good multilingual + small enough
for an 8GB RTX 4060 at Q4 quantization."""

import json

from pydantic import ValidationError

from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.schemas.vlm_output import VLMOutput, VLMParseError


class MiniCPMVProvider:
    """Implements `VLMProvider`. Tag in Ollama: `minicpm-v:8b`."""

    name = "minicpm-v-2.6"
    _model_tag = "minicpm-v:8b"

    def __init__(self, ollama: OllamaClient):
        self._ollama = ollama

    async def verify(
        self,
        frames: list[bytes],
        prompt: str,
        timeout_sec: int,
    ) -> VLMOutput:
        # timeout_sec is enforced by the httpx client constructed in OllamaClient
        del timeout_sec  # MiniCPM uses the client's default; OK for M1
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
