"""One-thread-per-camera worker.

Loop:
  1. cv2.VideoCapture(rtsp_url) — opens once, auto-reconnects on read failure
  2. read() each frame
  3. every Nth frame: YOLO → tracker → metadata payload → emitter
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque

import cv2

from sentry_ai.live_worker.emitter import MetadataEmitter
from sentry_ai.live_worker.schemas import FrameMetadata, TrackPayload
from sentry_ai.live_worker.tracker import ByteTrackWrapper
from sentry_ai.live_worker.yolo_runner import YoloPoseRunner
from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.camera")

# Rolling window for FPS smoothing
_FPS_WINDOW_SEC = 5.0


class CameraWorker:
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        emitter: MetadataEmitter,
        frame_skip: int = 3,
        yolo_conf: float = 0.35,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame_skip = max(1, frame_skip)
        self.yolo_conf = yolo_conf
        self._emitter = emitter
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Stats
        self._frames_total = 0
        self._detections_total = 0
        self._capture_times: deque[float] = deque(maxlen=200)   # for fps_capture
        self._inference_times: deque[float] = deque(maxlen=200)  # for fps_inference
        self._last_error: str | None = None

        # Lazy-init heavy components on first run (not on construct) so
        # `LiveStartRequest` validation can happen без CUDA cost
        self._yolo: YoloPoseRunner | None = None
        self._tracker: ByteTrackWrapper | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"live-cam-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        log.info("camera.started", camera_id=self.camera_id, rtsp_url=self.rtsp_url)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        log.info(
            "camera.stopped",
            camera_id=self.camera_id,
            frames_total=self._frames_total,
            detections_total=self._detections_total,
        )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fps_capture(self) -> float:
        return _fps_from_window(self._capture_times)

    @property
    def fps_inference(self) -> float:
        return _fps_from_window(self._inference_times)

    @property
    def frames_total(self) -> int:
        return self._frames_total

    @property
    def detections_total(self) -> int:
        return self._detections_total

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # === Worker loop ===

    def _run(self) -> None:
        # Lazy heavy init inside the worker thread (avoid blocking the API call)
        try:
            self._yolo = YoloPoseRunner(conf=self.yolo_conf)
            # ByteTrack frame_rate hint = effective post-skip FPS target
            self._tracker = ByteTrackWrapper(frame_rate=max(1, 30 // self.frame_skip))
        except Exception as e:
            self._last_error = f"init failed: {e}"
            log.exception("camera.init_failed", camera_id=self.camera_id)
            return

        cap: cv2.VideoCapture | None = None
        backoff = 1.0
        try:
            while not self._stop_event.is_set():
                if cap is None or not cap.isOpened():
                    cap = self._open_capture()
                    if cap is None:
                        # Sleep with stop check
                        if self._stop_event.wait(timeout=backoff):
                            break
                        backoff = min(backoff * 2, 30.0)
                        continue
                    backoff = 1.0

                ok, frame = cap.read()
                if not ok or frame is None:
                    log.warning("camera.read_failed", camera_id=self.camera_id)
                    cap.release()
                    cap = None
                    continue

                self._frames_total += 1
                self._capture_times.append(time.monotonic())

                if self._frames_total % self.frame_skip != 0:
                    continue

                try:
                    self._process_frame(frame)
                except Exception as e:  # noqa: BLE001
                    self._last_error = f"process failed: {e}"
                    log.exception(
                        "camera.process_failed",
                        camera_id=self.camera_id,
                    )
        finally:
            if cap is not None:
                cap.release()

    def _open_capture(self) -> cv2.VideoCapture | None:
        # ffmpeg backend, TCP transport for RTSP reliability
        # OpenCV honors OPENCV_FFMPEG_CAPTURE_OPTIONS env for low-level tuning
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self._last_error = "VideoCapture failed to open"
            log.warning("camera.open_failed", camera_id=self.camera_id, url=self.rtsp_url)
            return None
        # Smaller internal buffer → fresher frames, less latency
        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        log.info("camera.open_ok", camera_id=self.camera_id)
        return cap

    def _process_frame(self, frame_bgr) -> None:  # type: ignore[no-untyped-def]
        assert self._yolo is not None
        assert self._tracker is not None

        h, w = frame_bgr.shape[:2]
        detections = self._yolo.detect_persons(frame_bgr)
        tracked = self._tracker.update(detections)
        self._inference_times.append(time.monotonic())
        self._detections_total += len(detections)

        tracks_payload = [
            TrackPayload(
                person_id=t.tracker_id,
                box=t.box,
                det_confidence=t.score,
            )
            for t in tracked
        ]

        payload = FrameMetadata(
            camera_id=self.camera_id,
            frame_id=self._frames_total,
            ts_ms=int(time.time() * 1000),
            width=w,
            height=h,
            fps_inference=self.fps_inference,
            tracks=tracks_payload,
        )
        self._emitter.enqueue(payload)


def _fps_from_window(times: deque[float]) -> float:
    if len(times) < 2:
        return 0.0
    now = time.monotonic()
    recent = [t for t in times if now - t <= _FPS_WINDOW_SEC]
    if len(recent) < 2:
        return 0.0
    span = recent[-1] - recent[0]
    if span <= 0:
        return 0.0
    return (len(recent) - 1) / span
