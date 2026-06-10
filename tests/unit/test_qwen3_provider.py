"""Unit tests for Qwen3VLProvider — same contract as Qwen2.5-VL, new model tag."""

import json

import httpx
import pytest
import respx

from sentry_ai.providers.ollama_client import OllamaClient
from sentry_ai.providers.qwen3_vl import Qwen3VLProvider
from sentry_ai.schemas.vlm_output import Category, VLMParseError


@pytest.fixture
def ollama() -> OllamaClient:
    return OllamaClient(base_url="http://ollama.test", timeout_sec=5)


async def test_qwen3_parses_valid_response(ollama: OllamaClient) -> None:
    with respx.mock(base_url="http://ollama.test") as mock:
        route = mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "category": "pocket_conceal",
                                "confidence": 0.88,
                                "reasoning": "Хүн бараа авч халаас руу нуун, эргэж харсан.",
                            }
                        ),
                    }
                },
            )
        )
        provider = Qwen3VLProvider(ollama)
        out = await provider.verify(frames=[b"f1"], prompt="x", timeout_sec=5)
        assert out.category is Category.pocket_conceal
        assert out.confidence == 0.88
        # Verify the right model tag was sent to Ollama
        sent = route.calls[0].request.read()
        assert b"qwen3-vl:4b-instruct" in sent


async def test_qwen3_factory_resolves() -> None:
    """ADR-0026: qwen3-vl-4b joins the registry; qwen2.5-vl-7b stays for rollback."""
    from sentry_ai.providers.factory import list_provider_names

    names = list_provider_names()
    assert "minicpm-v-2.6" in names
    assert "qwen3-vl-4b" in names
    assert "qwen2.5-vl-7b" in names


async def test_qwen3_raises_on_malformed_json(ollama: OllamaClient) -> None:
    with respx.mock(base_url="http://ollama.test") as mock:
        mock.post("/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "totally garbage"}},
            )
        )
        provider = Qwen3VLProvider(ollama)
        with pytest.raises(VLMParseError):
            await provider.verify(frames=[b"x"], prompt="x", timeout_sec=5)
