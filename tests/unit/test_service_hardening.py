"""Service-token auth + SSRF/path input hardening (enforce-if-configured)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from sentry_ai import runtime
from sentry_ai.api.v1.live import _validate_rtsp_url
from sentry_ai.auth import require_service_token
from sentry_ai.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize("url", ["rtsp://media/cam1", "rtsps://media:8554/cam2"])
def test_rtsp_scheme_allowed(url: str) -> None:
    _validate_rtsp_url(url)  # does not raise


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://169.254.169.254/", "https://internal/", "ftp://x/y"]
)
def test_rtsp_scheme_rejected(url: str) -> None:
    with pytest.raises(HTTPException) as exc:
        _validate_rtsp_url(url)
    assert exc.value.status_code == 400


def test_railway_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    for m in runtime._RAILWAY_MARKERS:
        monkeypatch.delenv(m, raising=False)
    assert runtime.is_railway_host() is False
    monkeypatch.setenv("RAILWAY_SERVICE_ID", "svc_123")
    assert runtime.is_railway_host() is True


async def test_service_token_open_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_SERVICE_TOKEN", raising=False)
    get_settings.cache_clear()
    # No token configured → no header required (trusted-LAN default).
    await require_service_token(authorization=None)


async def test_service_token_enforced_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_SERVICE_TOKEN", "s3cret")
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as missing:
        await require_service_token(authorization=None)
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException):
        await require_service_token(authorization="Bearer wrong")
    # Correct token passes.
    await require_service_token(authorization="Bearer s3cret")
