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
# Live-breach topology pushed from the backend (ADR-0026 central control), the
# SINGLE authority for whether this node creates breach alerts. None until the
# first node-config poll, when the env bootstrap (settings.live_alert_push_enabled)
# is the fallback. "node_push" = detect + cut + VLM + POST; "off" = no alerts.
_BREACH_MODES = ("node_push", "off")
_central_breach_mode: str | None = None


@dataclass(frozen=True)
class ProviderHealth:
    effective: str | None  # provider the verify path will actually use
    ready: bool  # backing model reachable (Ollama tag pulled / vLLM up)
    error: str | None  # human-readable (Mongolian) reason when not ready


_health = ProviderHealth(effective=None, ready=False, error=None)


# === Per-node YOLO + scan/VLM tuning (central control) ===
# The node polls these from /api/v1/ai-nodes/config and hot-applies them WITHOUT a
# restart: the camera read-loop reads frame_skip live, _process_frame syncs the
# YOLO conf + item cadence + scan interval each frame, and the verify path reads
# frames_per_clip/frame_max_dim per breach. Any field left None means "the call
# site keeps its own default" (the node's .env / hardcoded value), so an older
# backend that doesn't send these keys changes nothing.


@dataclass(frozen=True)
class DetectionConfig:
    frame_skip: int | None = None
    person_conf: float | None = None
    item_conf: float | None = None
    item_every_n: int | None = None
    scan_interval_sec: float | None = None
    frames_per_clip: int | None = None
    frame_max_dim: int | None = None


_detection: DetectionConfig | None = None


def _as_int(v: object) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _as_float(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def set_central_detection(cfg: dict[str, object] | None) -> bool:
    """Called by the config poller after each node-config fetch. Coerces the raw
    JSON into a DetectionConfig, dropping any non-numeric value so a bad DB entry
    can never crash the hot path. Returns True if the effective config CHANGED
    (so the caller can log it once)."""
    global _detection
    if not isinstance(cfg, dict):
        return False
    new = DetectionConfig(
        frame_skip=_as_int(cfg.get("frame_skip")),
        person_conf=_as_float(cfg.get("person_conf")),
        item_conf=_as_float(cfg.get("item_conf")),
        item_every_n=_as_int(cfg.get("item_every_n")),
        scan_interval_sec=_as_float(cfg.get("scan_interval_sec")),
        frames_per_clip=_as_int(cfg.get("frames_per_clip")),
        frame_max_dim=_as_int(cfg.get("frame_max_dim")),
    )
    with _lock:
        changed = new != _detection
        _detection = new
    return changed


def get_detection() -> DetectionConfig | None:
    with _lock:
        return _detection


def resolve_frames_per_clip(default: int) -> int:
    """Keyframes per VLM scan — the central per-node value if set, else the
    .env/settings default. Read by the verify path on each breach."""
    cfg = get_detection()
    if cfg is not None and cfg.frames_per_clip is not None:
        return max(1, cfg.frames_per_clip)
    return default


def resolve_frame_max_dim(default: int) -> int:
    """Max keyframe edge (px) sent to the VLM — central per-node value or default."""
    cfg = get_detection()
    if cfg is not None and cfg.frame_max_dim is not None:
        return max(64, cfg.frame_max_dim)
    return default


def set_central_provider(name: str | None) -> None:
    """Called by the config poller after each successful node-config fetch."""
    global _central_provider
    with _lock:
        _central_provider = name


def get_central_provider() -> str | None:
    with _lock:
        return _central_provider


def set_central_breach_mode(mode: str | None) -> None:
    """Called by the config poller after each node-config fetch. Ignores unknown
    values so a typo in the DB never silently disables alerting."""
    global _central_breach_mode
    if mode is not None and mode not in _BREACH_MODES:
        return
    with _lock:
        _central_breach_mode = mode


def resolve_breach_mode() -> str:
    """The breach topology this node will actually apply: the centrally-chosen
    value if one has been polled, else the .env bootstrap
    (live_alert_push_enabled → node_push/off). One source → the node and backend
    can never drift into double-firing or a silent no-op."""
    with _lock:
        central = _central_breach_mode
    if central in _BREACH_MODES:
        return central
    from sentry_ai.settings import get_settings

    return "node_push" if get_settings().live_alert_push_enabled else "off"


def set_provider_health(effective: str | None, ready: bool, error: str | None) -> None:
    """Called by the config poller after resolving + readiness-checking the
    effective provider."""
    global _health
    with _lock:
        _health = ProviderHealth(effective=effective, ready=ready, error=error)


# --- VLM verify activity (so the dashboard can show the VLM HAS run on the GPU,
# even though it's event-driven and idle most of the time). Recorded by every
# verify path via pipeline.verifier.verify_clip; read by the /v1/live/provider
# endpoint (same process) and reported in the heartbeat.
import time  # noqa: E402

_vlm_count = 0
_vlm_last_at: float | None = None
_vlm_last_latency_ms: int | None = None


def record_vlm_verify(latency_ms: int) -> None:
    global _vlm_count, _vlm_last_at, _vlm_last_latency_ms
    with _lock:
        _vlm_count += 1
        _vlm_last_at = time.time()
        _vlm_last_latency_ms = latency_ms


def get_vlm_activity() -> dict[str, object]:
    """{count, last_ago_sec, last_latency_ms}. last_ago_sec computed in-process so
    it never depends on the browser's clock matching the node's."""
    with _lock:
        count = _vlm_count
        last_at = _vlm_last_at
        lat = _vlm_last_latency_ms
    ago = int(time.time() - last_at) if last_at is not None else None
    return {"count": count, "last_ago_sec": ago, "last_latency_ms": lat}


def get_provider_health() -> ProviderHealth:
    with _lock:
        return _health
