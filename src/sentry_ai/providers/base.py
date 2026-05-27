"""VLM provider Protocol — pluggable backends behind a uniform interface."""

from typing import Protocol

from sentry_ai.schemas.vlm_output import VLMOutput


class VLMProvider(Protocol):
    """Implementations: MiniCPMVProvider, QwenVLProvider, ..."""

    name: str
    """The model identifier reported in Alert.model_name."""

    async def verify(
        self,
        frames: list[bytes],
        prompt: str,
        timeout_sec: int,
    ) -> VLMOutput:
        """Send frames (JPEG bytes) + prompt to the model.

        Must return a parsed VLMOutput or raise VLMParseError on malformed JSON.
        Network/timeout errors should propagate as httpx exceptions.
        """
        ...
