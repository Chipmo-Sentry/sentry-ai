"""Periodically fetch behavior config from sentry-backend.

GET /api/v1/behaviors → {dimensions:[{key, weight, ...}], thresholds:{green_max, yellow_max}}.
Updates a shared dict that BehaviorScorer instances consult on each frame
(via update_weights / update_thresholds called by the manager).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import httpx

from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.live_worker.config_poller")

POLL_INTERVAL_SEC = 30.0
REQUEST_TIMEOUT_SEC = 5.0


def _num_map(raw: Any) -> dict[str, float]:
    """Coerce a JSON object into a {str: float} map, dropping non-numeric values."""
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(v)
    return out


class BehaviorConfigPoller:
    """Singleton-ish: started by manager.start_camera() on first invocation."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Callbacks invoked on each successful refresh:
        #   on_weights(dict[str, float])
        #   on_thresholds(green_max, yellow_max, high_max)
        self._on_weights: list[Callable[[dict[str, float]], None]] = []
        self._on_thresholds: list[Callable[[float, float, float], None]] = []
        # on_params(engine: dict, detector: dict[key, dict])
        self._on_params: list[Callable[[dict[str, float], dict[str, dict[str, float]]], None]] = []
        self._last_weights: dict[str, float] | None = None
        self._last_thresholds: tuple[float, float, float] | None = None
        self._last_params: tuple[dict[str, float], dict[str, dict[str, float]]] | None = None
        self._lock = threading.Lock()
        # Debounce TRANSIENT provider-readiness failures (Ollama momentarily busy /
        # mid-model-load) so a single slow probe doesn't flap the dashboard to a
        # scary "Ollama холбогдсонгүй". A real outage still surfaces after a few polls.
        self._provider_fail_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="behavior-config-poller",
            daemon=True,
        )
        self._thread.start()
        log.info("config_poller.started", interval=POLL_INTERVAL_SEC)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def subscribe(
        self,
        on_weights: Callable[[dict[str, float]], None],
        on_thresholds: Callable[[float, float, float], None],
        on_params: Callable[[dict[str, float], dict[str, dict[str, float]]], None] | None = None,
    ) -> None:
        """Register callbacks; immediately invoked with last-known config if any."""
        with self._lock:
            self._on_weights.append(on_weights)
            self._on_thresholds.append(on_thresholds)
            if on_params is not None:
                self._on_params.append(on_params)
            if self._last_weights is not None:
                on_weights(self._last_weights)
            if self._last_thresholds is not None:
                on_thresholds(*self._last_thresholds)
            if on_params is not None and self._last_params is not None:
                on_params(*self._last_params)

    def _run(self) -> None:
        settings = get_settings()
        url = settings.sentry_backend_url.rstrip("/") + "/api/v1/behaviors"
        # First fetch fast; then poll on interval
        delay = 1.0
        while not self._stop.is_set():
            try:
                self._fetch_and_dispatch(url)
                delay = POLL_INTERVAL_SEC
            except (httpx.HTTPError, ValueError) as e:
                # Don't spam — log every failed poll at warning only
                log.warning("config_poller.fetch_failed", error=str(e), retry_in=delay)
                delay = min(delay * 2, POLL_INTERVAL_SEC)
            # Central VLM provider (ADR-0026 central control). Independent of the
            # behavior poll above so a node-config failure never stalls behavior.
            # Broad guard: nothing in here may ever kill the poller thread.
            try:
                self._fetch_node_provider(settings)
            except Exception:  # noqa: BLE001 — poller must survive any provider error
                log.warning("config_poller.node_provider_unexpected", exc_info=True)
            if self._stop.wait(timeout=delay):
                break

    def _fetch_node_provider(self, settings: Any) -> None:
        """Poll /api/v1/ai-nodes/config for the centrally-chosen VLM provider,
        publish it to runtime_config so the verify path hot-applies it, then
        readiness-check the effective provider so the dashboard can show whether
        the server actually applied it. Best-effort, paired nodes only; failures
        are logged and never raised."""
        from sentry_ai.providers.factory import resolve_provider_name
        from sentry_ai.runtime_config import (
            set_central_breach_mode,
            set_central_provider,
            set_provider_health,
        )

        if not settings.ai_node_id:
            return
        url = settings.sentry_backend_url.rstrip("/") + "/api/v1/ai-nodes/config"
        headers = {"Authorization": f"Bearer {settings.sentry_backend_service_token}"}
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT_SEC) as client:
                r = client.get(url, headers=headers)
                r.raise_for_status()
                cfg = r.json()
                provider = cfg.get("provider")
                breach_mode = cfg.get("breach_mode")
        except (httpx.HTTPError, ValueError) as e:
            log.warning("config_poller.node_config_failed", error=str(e))
            return
        if isinstance(provider, str) and provider:
            set_central_provider(provider)
        # Live-breach topology (central control). set_central_breach_mode ignores
        # unknown values, so a bad DB value can't silently disable alerting.
        if isinstance(breach_mode, str) and breach_mode:
            set_central_breach_mode(breach_mode)
        # Readiness: which provider will verify actually use, and is its model up?
        effective = resolve_provider_name(None)
        ready, error, transient = _check_provider_ready(effective, settings)
        if ready or not transient:
            # ready, or a deterministic config error (model not pulled) → report now.
            self._provider_fail_count = 0
            set_provider_health(effective, ready, error)
        else:
            # Transient (runtime momentarily unreachable). Show a neutral "checking"
            # state, not a red error, until it fails PROVIDER_FAIL_LIMIT polls in a row.
            self._provider_fail_count += 1
            if self._provider_fail_count >= PROVIDER_FAIL_LIMIT:
                set_provider_health(effective, False, error)
            else:
                set_provider_health(effective, False, None)

    def _fetch_and_dispatch(self, url: str) -> None:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SEC) as client:
            r = client.get(url)
            r.raise_for_status()
            data: dict[str, Any] = r.json()

        # Disabled criteria (active=false) contribute 0 — the scorer needs no
        # change since it just reads weights[key]. Custom criteria without a
        # coded detector are simply never looked up by the scorer.
        weights = {
            d["key"]: (float(d["weight"]) if d.get("active", True) else 0.0)
            for d in data.get("dimensions", [])
            if "key" in d and "weight" in d
        }
        thresholds = data.get("thresholds", {})
        # v2 absolute 0-100 defaults (ADR-0024): LOW<10, MEDIUM<25, HIGH<50.
        green_max = float(thresholds.get("green_max", 10.0))
        yellow_max = float(thresholds.get("yellow_max", 25.0))
        high_max = float(thresholds.get("high_max", 50.0))
        triple = (green_max, yellow_max, high_max)

        # Engine globals + per-detector sensitivity params (ADR-0024 v2 tuning).
        # `engine` is a flat float map; each dimension may carry a `params` object.
        engine = _num_map(data.get("engine", {}))
        detector: dict[str, dict[str, float]] = {}
        for d in data.get("dimensions", []):
            key = d.get("key")
            params = d.get("params")
            if isinstance(key, str) and isinstance(params, dict):
                p = _num_map(params)
                if p:
                    detector[key] = p
        params_pair = (engine, detector)

        with self._lock:
            weights_changed = self._last_weights != weights
            thr_changed = self._last_thresholds != triple
            params_changed = self._last_params != params_pair
            self._last_weights = weights
            self._last_thresholds = triple
            self._last_params = params_pair
            wcbs = list(self._on_weights)
            tcbs = list(self._on_thresholds)
            pcbs = list(self._on_params)

        if weights_changed:
            for wcb in wcbs:
                try:
                    wcb(weights)
                except Exception:  # noqa: BLE001
                    log.exception("config_poller.weights_cb_failed")
        if thr_changed:
            for tcb in tcbs:
                try:
                    tcb(green_max, yellow_max, high_max)
                except Exception:  # noqa: BLE001
                    log.exception("config_poller.thresholds_cb_failed")
        if params_changed:
            for pcb in pcbs:
                try:
                    pcb(engine, detector)
                except Exception:  # noqa: BLE001
                    log.exception("config_poller.params_cb_failed")
        if weights_changed or thr_changed or params_changed:
            log.info(
                "config_poller.refreshed",
                weights_changed=weights_changed,
                thr_changed=thr_changed,
                green_max=green_max,
                yellow_max=yellow_max,
                high_max=high_max,
            )


PROVIDER_FAIL_LIMIT = 3  # consecutive transient probe failures before alarming
_PROBE_TIMEOUT_SEC = 8.0  # generous — Ollama can be slow while loading a model


def _get_with_retry(url: str) -> httpx.Response:
    """GET with one retry — absorbs a single slow/blipped response (e.g. Ollama
    busy mid-model-load) so readiness doesn't false-fail on a transient hiccup."""
    last: httpx.HTTPError | None = None
    for _ in range(2):
        try:
            with httpx.Client(timeout=_PROBE_TIMEOUT_SEC) as c:
                resp = c.get(url)
                resp.raise_for_status()
                return resp
        except httpx.HTTPError as e:
            last = e
    raise last if last is not None else httpx.HTTPError("probe failed")


def _check_provider_ready(effective: str, settings: Any) -> tuple[bool, str | None, bool]:
    """Confirm the effective provider's backing model is reachable, so the dashboard
    surfaces a real error (e.g. model not pulled) instead of failing only at the next
    breach. Returns (ready, error_message_mn, transient). `transient=True` means a
    momentary connectivity blip the caller should debounce; `transient=False` means a
    deterministic config error (model not pulled) safe to surface immediately. Never raises."""
    from sentry_ai.providers.factory import get_provider_class

    cls = get_provider_class(effective)
    if cls is None:
        return False, f"Тодорхойгүй provider: {effective}", False
    runtime = getattr(cls, "runtime", "ollama")
    try:
        if runtime == "vllm":
            base = settings.vllm_base_url.rstrip("/")
            _get_with_retry(base + "/models")
            return True, None, False
        # Ollama-backed: the model tag must be pulled on this node.
        tag = getattr(cls, "model_tag", "")
        r = _get_with_retry(settings.ollama_base_url.rstrip("/") + "/api/tags")
        names = {m.get("name", "") for m in r.json().get("models", [])}
        if tag and tag not in names:
            return False, f"Ollama-д '{tag}' татагдаагүй — `ollama pull {tag}`", False
    except httpx.HTTPError:
        where = "vLLM сервер" if runtime == "vllm" else "Ollama"
        return False, f"{where} холбогдсонгүй", True  # transient → caller debounces
    return True, None, False


_poller: BehaviorConfigPoller | None = None


def get_config_poller() -> BehaviorConfigPoller:
    global _poller
    if _poller is None:
        _poller = BehaviorConfigPoller()
    return _poller
