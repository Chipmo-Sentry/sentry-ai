"""Runtime config pushed from the backend (central control) + provider health.

Holds the AI-node config the node polls from `/api/v1/ai-nodes/config` — today
just the centrally-chosen VLM `provider`. Stored here (process-global, thread
-safe) so the verify path can pick it up live, without a restart, while the
config poller (a worker thread in the same process) keeps it fresh.

Authority model: an explicit per-request `provider` wins; else the central value
(if set and a known provider) wins; else `settings.default_provider` (.env) is
the bootstrap fallback. See `providers.factory.resolve_provider_name`.

Also tracks PROVIDER HEALTH — the provider the node will actually use
(`effective`) plus whether its backing model is reachable (`ready` + `error`),
set by the config poller after each readiness check. The heartbeat reports this
so the dashboard can show "applied on the server" vs "applying…" vs an error.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

_lock = threading.Lock()
_central_provider: str | None = None


@dataclass(frozen=True)
class ProviderHealth:
    effective: str | None  # provider the verify path will actually use
    ready: bool  # backing model reachable (Ollama tag pulled / vLLM up)
    error: str | None  # human-readable (Mongolian) reason when not ready


_health = ProviderHealth(effective=None, ready=False, error=None)


def set_central_provider(name: str | None) -> None:
    """Called by the config poller after each successful node-config fetch."""
    global _central_provider
    with _lock:
        _central_provider = name


def get_central_provider() -> str | None:
    with _lock:
        return _central_provider


def set_provider_health(effective: str | None, ready: bool, error: str | None) -> None:
    """Called by the config poller after resolving + readiness-checking the
    effective provider."""
    global _health
    with _lock:
        _health = ProviderHealth(effective=effective, ready=ready, error=error)


def get_provider_health() -> ProviderHealth:
    with _lock:
        return _health
