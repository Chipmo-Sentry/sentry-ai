"""Service-token auth for /v1/* routes (enforce-if-configured).

When ``AI_SERVICE_TOKEN`` is set, every protected route requires
``Authorization: Bearer <token>``. When it is unset/empty the dependency is a
no-op so the trusted-LAN M1 deployment keeps working without a token.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from sentry_ai.settings import get_settings


async def require_service_token(authorization: str | None = Header(default=None)) -> None:
    """Reject requests lacking a valid bearer token — unless no token is set."""
    token = get_settings().ai_service_token
    if not token:
        # Not configured → open (trusted-LAN default). Production MUST set it.
        return
    expected = f"Bearer {token}"
    # Constant-time compare; reject missing/malformed headers.
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing service token",
            headers={"WWW-Authenticate": "Bearer"},
        )
