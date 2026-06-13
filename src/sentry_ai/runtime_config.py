"""Runtime config pushed from the backend (central control).

Holds the AI-node config the node polls from `/api/v1/ai-nodes/config` — today
just the centrally-chosen VLM `provider`. Stored here (process-global, thread
-safe) so the verify path can pick it up live, without a restart, while the
config poller (a worker thread in the same process) keeps it fresh.

Authority model: an explicit per-request `provider` wins; else this central
value (if set and a known provider) wins; else `settings.default_provider`
(.env) is the bootstrap fallback. See `providers.factory.resolve_provider_name`.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_central_provider: str | None = None


def set_central_provider(name: str | None) -> None:
    """Called by the config poller after each successful node-config fetch."""
    global _central_provider
    with _lock:
        _central_provider = name


def get_central_provider() -> str | None:
    with _lock:
        return _central_provider
