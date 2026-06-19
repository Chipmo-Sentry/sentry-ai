"""Unit tests for MiniCPMVProvider (Ollama mocked via respx)."""

import json

import httpx
import pytest
import respx

from sentry_ai.providers.minicpm_v import MiniCPMVProvider
from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.schemas.vlm_output import Category, VLMParseError


@pytest.fixture
def ollama() -> OllamaClient:
    return OllamaClient(base_url="http://ollama.test", timeout_sec=5)


async def test_provider_parses_valid_response(ollama: OllamaClient) -> None:
    with respx.mock(base_url="http://ollama.test") as mock:
        mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "actions": ["cart_pickup"],
                                "confidence": 0.72,
                                "reasoning": "Бараа авч сагсанд тавьсан, эргэж харсан.",
                            }
                        ),
                    }
                },
            )
        )
        provider = MiniCPMVProvider(ollama)
        out = await provider.verify(
            frames=[b"fakejpeg1", b"fakejpeg2"],
            prompt="test prompt",
            timeout_sec=5,
        )
        assert out.category is Category.cart_pickup
        assert out.confidence == 0.72


async def test_provider_raises_on_malformed_json(ollama: OllamaClient) -> None:
    with respx.mock(base_url="http://ollama.test") as mock:
        mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "not json at all"}},
            )
        )
        provider = MiniCPMVProvider(ollama)
        with pytest.raises(VLMParseError):
            await provider.verify(frames=[b"fake"], prompt="x", timeout_sec=5)


async def test_provider_raises_on_invalid_schema(ollama: OllamaClient) -> None:
    with respx.mock(base_url="http://ollama.test") as mock:
        mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"actions": ["INVALID"], "confidence": 0.5, "reasoning": "ok"}
                        ),
                    }
                },
            )
        )
        provider = MiniCPMVProvider(ollama)
        with pytest.raises(VLMParseError):
            await provider.verify(frames=[b"x"], prompt="x", timeout_sec=5)
