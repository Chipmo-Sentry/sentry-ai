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


class BehaviorConfigPoller:
    """Singleton-ish: started by manager.start_camera() on first invocation."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Callbacks invoked on each successful refresh:
        #   on_weights(dict[str, float])
        #   on_thresholds(green_max, yellow_max)
        self._on_weights: list[Callable[[dict[str, float]], None]] = []
        self._on_thresholds: list[Callable[[float, float], None]] = []
        self._last_weights: dict[str, float] | None = None
        self._last_thresholds: tuple[float, float] | None = None
        self._lock = threading.Lock()

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
        on_thresholds: Callable[[float, float], None],
    ) -> None:
        """Register callbacks; immediately invoked with last-known config if any."""
        with self._lock:
            self._on_weights.append(on_weights)
            self._on_thresholds.append(on_thresholds)
            if self._last_weights is not None:
                on_weights(self._last_weights)
            if self._last_thresholds is not None:
                on_thresholds(*self._last_thresholds)

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
            if self._stop.wait(timeout=delay):
                break

    def _fetch_and_dispatch(self, url: str) -> None:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SEC) as client:
            r = client.get(url)
            r.raise_for_status()
            data: dict[str, Any] = r.json()

        weights = {
            d["key"]: float(d["weight"])
            for d in data.get("dimensions", [])
            if "key" in d and "weight" in d
        }
        thresholds = data.get("thresholds", {})
        green_max = float(thresholds.get("green_max", 5.0))
        yellow_max = float(thresholds.get("yellow_max", 15.0))

        with self._lock:
            weights_changed = self._last_weights != weights
            thr_changed = self._last_thresholds != (green_max, yellow_max)
            self._last_weights = weights
            self._last_thresholds = (green_max, yellow_max)
            wcbs = list(self._on_weights)
            tcbs = list(self._on_thresholds)

        if weights_changed:
            for wcb in wcbs:
                try:
                    wcb(weights)
                except Exception:  # noqa: BLE001
                    log.exception("config_poller.weights_cb_failed")
        if thr_changed:
            for tcb in tcbs:
                try:
                    tcb(green_max, yellow_max)
                except Exception:  # noqa: BLE001
                    log.exception("config_poller.thresholds_cb_failed")
        if weights_changed or thr_changed:
            log.info(
                "config_poller.refreshed",
                weights_changed=weights_changed,
                thr_changed=thr_changed,
                green_max=green_max,
                yellow_max=yellow_max,
            )


_poller: BehaviorConfigPoller | None = None


def get_config_poller() -> BehaviorConfigPoller:
    global _poller
    if _poller is None:
        _poller = BehaviorConfigPoller()
    return _poller
