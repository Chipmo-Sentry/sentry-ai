"""Unit tests for Qwen3VLvLLMProvider — OpenAI-compatible vLLM endpoint."""

import json

import httpx
import pytest
import respx

from sentry_ai.providers.vllm_qwen3 import Qwen3VLvLLMProvider
from sentry_ai.schemas.vlm_output import Category, VLMParseError
from sentry_ai.settings import get_settings


@pytest.fixture(autouse=True)
def _point_at_test_server() -> None:
    # vllm_base_url defaults to localhost; the provider reads it at construct time.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


async def test_vllm_parses_valid_response() -> None:
    base = get_settings().vllm_base_url
    with respx.mock(base_url=base) as mock:
        route = mock.post("/chat/completions").mock(
            return_value=_chat_response(
                json.dumps(
                    {
                        "actions": ["pocket_conceal"],
                        "confidence": 0.87,
                        "reasoning": "Хүн бараа авч халаас руу нуун, эргэж харсан.",
                    }
                )
            )
        )
        provider = Qwen3VLvLLMProvider(ollama=None)  # type: ignore[arg-type]
        out = await provider.verify(frames=[b"f1", b"f2"], prompt="x", timeout_sec=5)
        assert out.category is Category.pocket_conceal
        assert out.confidence == 0.87
        sent = route.calls[0].request.read()
        # served model id + an image_url data URI must be present
        assert b"Qwen/Qwen3-VL-4B-Instruct" in sent
        assert b"data:image/jpeg;base64," in sent
        assert b"json_object" in sent


async def test_vllm_factory_resolves() -> None:
    from sentry_ai.providers.factory import list_provider_names

    assert "qwen3-vl-vllm" in list_provider_names()


async def test_vllm_raises_on_malformed_json() -> None:
    base = get_settings().vllm_base_url
    with respx.mock(base_url=base) as mock:
        mock.post("/chat/completions").mock(return_value=_chat_response("totally garbage"))
        provider = Qwen3VLvLLMProvider(ollama=None)  # type: ignore[arg-type]
        with pytest.raises(VLMParseError):
            await provider.verify(frames=[b"x"], prompt="x", timeout_sec=5)


async def test_vllm_raises_on_unexpected_shape() -> None:
    base = get_settings().vllm_base_url
    with respx.mock(base_url=base) as mock:
        mock.post("/chat/completions").mock(return_value=httpx.Response(200, json={"oops": 1}))
        provider = Qwen3VLvLLMProvider(ollama=None)  # type: ignore[arg-type]
        with pytest.raises(VLMParseError):
            await provider.verify(frames=[b"x"], prompt="x", timeout_sec=5)
