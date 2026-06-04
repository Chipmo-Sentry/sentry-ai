"""Runtime host detection.

sentry-ai runs RTSP ingest + GPU inference and must NEVER run on Railway
(red-line #2 / ADR-0007). Railway injects RAILWAY_* env vars into every
container, so their presence is a reliable signal we are on a forbidden host.
"""

from __future__ import annotations

import os

# Set by Railway in every deployed container.
_RAILWAY_MARKERS = ("RAILWAY_ENVIRONMENT", "RAILWAY_SERVICE_ID", "RAILWAY_PROJECT_ID")


def is_railway_host() -> bool:
    """True when running inside a Railway container."""
    return any(os.environ.get(m) for m in _RAILWAY_MARKERS)
