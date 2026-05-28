"""Singleton manager — spawns + supervises CameraWorker instances."""

from __future__ import annotations

import threading
from functools import lru_cache

from sentry_ai.live_worker.camera_worker import CameraWorker
from sentry_ai.live_worker.emitter import MetadataEmitter
from sentry_ai.live_worker.schemas import LiveWorkerStatus
from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.manager")


class LiveWorkerManager:
    def __init__(self) -> None:
        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()
        self._emitter = MetadataEmitter()
        self._emitter_started = False

    def start_camera(self, camera_id: str, rtsp_url: str, frame_skip: int = 3) -> None:
        with self._lock:
            if not self._emitter_started:
                self._emitter.start()
                self._emitter_started = True

            existing = self._workers.get(camera_id)
            if existing is not None and existing.running:
                if existing.rtsp_url == rtsp_url:
                    log.info("manager.already_running", camera_id=camera_id)
                    return
                # URL changed → restart
                existing.stop()

            worker = CameraWorker(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                emitter=self._emitter,
                frame_skip=frame_skip,
            )
            worker.start()
            self._workers[camera_id] = worker

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(camera_id)
            if worker is None:
                return False
            worker.stop()
            return True

    def stop_all(self) -> None:
        with self._lock:
            for w in self._workers.values():
                w.stop()
            self._workers.clear()
            if self._emitter_started:
                self._emitter.stop()
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
                    last_error=w.last_error,
                )
                for w in self._workers.values()
            ]

    @property
    def emitter_stats(self) -> dict[str, int]:
        return self._emitter.stats


@lru_cache(maxsize=1)
def get_manager() -> LiveWorkerManager:
    return LiveWorkerManager()
