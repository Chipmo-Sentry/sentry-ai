"""Service-token auth for /v1/* routes (enforce-if-configured).

When ``AI_SERVICE_TOKEN`` is set, every protected route requires
``Authorization: Bearer <token>``. When it is unset/empty the dependency is a
no-op so the trusted-LAN M1 deployment keeps working without a token.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from sentry_ai.settings import Settings, get_settings


def assert_service_token_configured(settings: Settings) -> None:
    """Refuse to start auth-open on a non-dev host (ADR-0029 §12 / I6).

    ``require_service_token`` below is a no-op when ``ai_service_token`` is
    empty, so a publicly-reachable verdict node (e.g. the rented vast GPU box)
    would accept UNAUTHENTICATED ``/v1/*`` calls. We therefore make the token
    mandatory whenever ``environment`` is ``staging`` or ``production``; ``dev``
    (trusted-LAN M1) stays open. Called once at startup — raising here makes
    uvicorn fail loudly rather than silently exposing the node.
    """
    if settings.environment in ("staging", "production") and not settings.ai_service_token:
        raise RuntimeError(
            "AI_SERVICE_TOKEN is required when ENVIRONMENT is staging/production "
            "(set it to the SAME value as the backend's SENTRY_AI_SERVICE_TOKEN). "
            "Refusing to start auth-open on a non-dev host."
        )


async def require_service_token(authorization: str | None = Header(default=None)) -> None:
    """Reject requests lacking a valid bearer token — unless no token is set."""
    token = get_settings().ai_service_token
    if not token:
        # Not configured → open (trusted-LAN default). Production MUST set it
        # (enforced at startup by assert_service_token_configured).
        return
    expected = f"Bearer {token}"
    # Constant-time compare; reject missing/malformed headers.
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing service token",
            headers={"WWW-Authenticate": "Bearer"},
        )
