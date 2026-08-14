"""Singleton manager — spawns + supervises CameraWorker instances."""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import lru_cache

from sentry_ai.live_worker import state_store
from sentry_ai.live_worker.camera_worker import CameraWorker
from sentry_ai.live_worker.config_poller import get_config_poller
from sentry_ai.live_worker.emitter import MetadataEmitter
from sentry_ai.live_worker.reid import StorePersonRegistry, make_embedder
from sentry_ai.live_worker.schemas import LiveWorkerStatus
from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.live_worker.manager")


class LiveWorkerManager:
    def __init__(self) -> None:
        self._workers: dict[str, CameraWorker] = {}
        # Per-camera config-poller unsubscribe handles (docs/33 Sprint C) — must
        # be called on stop/restart or the poller leaks dead workers' callbacks.
        self._unsubs: dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self._emitter = MetadataEmitter()
        self._emitter_started = False
        # One re-ID registry per store (ADR-0023 store-affinity: a store's cameras
        # all run on THIS node, so in-process is sufficient). Shared embedder.
        self._registries: dict[str, StorePersonRegistry] = {}
        self._embedder = make_embedder(get_settings().reid_model)

    def _get_registry(self, store_id: str) -> StorePersonRegistry:
        reg = self._registries.get(store_id)
        if reg is None:
            reg = StorePersonRegistry()
            self._registries[store_id] = reg
        return reg

    def start_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        frame_skip: int = 3,
        store_id: str | None = None,
        alert_threshold_pct: float | None = None,
        zones: list[dict[str, object]] | None = None,
    ) -> None:
        with self._lock:
            if not self._emitter_started:
                self._emitter.start()
                get_config_poller().start()
                self._emitter_started = True

            existing = self._workers.get(camera_id)
            if existing is not None and existing.running:
                # Restart on a changed source, breach threshold, OR zones so a
                # web-UI edit takes effect (the worker reads all three at start;
                # docs/29 P1c — a zone edit must restart to re-arm the engine).
                if (
                    existing.rtsp_url == rtsp_url
                    and existing.alert_threshold_pct == alert_threshold_pct
                    and existing.zones == (zones or None)
                ):
                    log.info("manager.already_running", camera_id=camera_id)
                    return
                existing.stop()
                # Drop the old worker's poller callbacks BEFORE re-subscribing —
                # otherwise every restart leaks a set (docs/33 Sprint C).
                self._unsubs.pop(camera_id, lambda: None)()

            registry = self._get_registry(store_id) if store_id else None
            worker = CameraWorker(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                emitter=self._emitter,
                frame_skip=frame_skip,
                store_id=store_id,
                registry=registry,
                embedder=self._embedder if registry is not None else None,
                alert_threshold_pct=alert_threshold_pct,
                zones=zones,
            )
            worker.start()
            self._workers[camera_id] = worker

            # Subscribe worker's scorer to live config updates (callbacks invoke
            # synchronously on the poller thread — both methods are thread-safe).
            # Keep the unsubscribe handle: it MUST run when this worker stops or
            # is replaced, or the poller accumulates dead workers forever.
            self._unsubs[camera_id] = get_config_poller().subscribe(
                on_weights=worker.apply_weights,
                on_thresholds=worker.apply_thresholds,
                on_params=worker.apply_params,
            )

            # Persist the spec so a node restart replays this worker (main.py
            # startup) — without this, a reboot leaves AI dark until the next
            # backend provision (bit us 2026-08-11).
            state_store.record_started(
                camera_id,
                {
                    "rtsp_url": rtsp_url,
                    "frame_skip": frame_skip,
                    "store_id": store_id,
                    "alert_threshold_pct": alert_threshold_pct,
                    "zones": zones,
                },
            )

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            # Pop (not just stop) so an intentionally stopped camera disappears
            # from /v1/live/status instead of lingering as a dead "error" worker
            # — which would false-alarm the per-camera health watchdog (T12 #3).
            worker = self._workers.pop(camera_id, None)
            self._unsubs.pop(camera_id, lambda: None)()
            # Deliberate deprovision — forget it even if no live worker matched,
            # so a stale saved spec can't resurrect a removed camera on reboot.
            state_store.record_stopped(camera_id)
            if worker is None:
                return False
            worker.stop()
            return True

    def stop_all(self) -> None:
        with self._lock:
            for w in self._workers.values():
                w.stop()
            self._workers.clear()
            for unsub in self._unsubs.values():
                unsub()
            self._unsubs.clear()
            if self._emitter_started:
                self._emitter.stop()
                get_config_poller().stop()
                self._emitter_started = False

    def status(self) -> list[LiveWorkerStatus]:
        with self._lock:
            return [
                LiveWorkerStatus(
                    camera_id=w.camera_id,
                    rtsp_url=w.rtsp_url,
                    running=w.running,
                    fps_capture=round(w.fps_capture, 2),
                    fps_inference=round(w.fps_inference, 2),
                    frames_total=w.frames_total,
                    detections_total=w.detections_total,
                    frame_skip=w.effective_frame_skip,
                    persons=w.persons_this_frame,
                    last_error=w.last_error,
                )
                for w in self._workers.values()
            ]

    @property
    def emitter_stats(self) -> dict[str, int]:
        return self._emitter.stats

    def snapshot(self, camera_id: str) -> tuple[bytes, float] | None:
        with self._lock:
            worker = self._workers.get(camera_id)
            if worker is None:
                return None
            return worker.latest_snapshot()


@lru_cache(maxsize=1)
def get_manager() -> LiveWorkerManager:
    return LiveWorkerManager()
