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
import numpy as np
from numpy.typing import NDArray

from sentry_ai.live_worker.behavior import BehaviorScorer
from sentry_ai.live_worker.emitter import MetadataEmitter
from sentry_ai.live_worker.reid import Embedder, StorePersonRegistry
from sentry_ai.live_worker.schemas import FrameMetadata, TrackPayload
from sentry_ai.live_worker.tracker import ByteTrackWrapper, TrackedDetection
from sentry_ai.live_worker.yolo_det import Item, YoloItemRunner
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
        store_id: str | None = None,
        registry: StorePersonRegistry | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame_skip = max(1, frame_skip)
        self.yolo_conf = yolo_conf
        self._emitter = emitter
        # Cross-camera re-ID (ADR-0022/0023): when a store registry + embedder are
        # provided, each person is linked to a store-global id and their risk
        # accumulates across the store's cameras. None → per-camera only.
        self.store_id = store_id
        self._registry = registry
        self._embedder = embedder
        self._prev_raw: dict[int, float] = {}  # per-track last raw score (for deltas)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Stats
        self._frames_total = 0
        self._detections_total = 0
        self._capture_times: deque[float] = deque(maxlen=200)  # for fps_capture
        self._inference_times: deque[float] = deque(maxlen=200)  # for fps_inference
        self._last_error: str | None = None

        # Lazy-init heavy components on first run (not on construct) so
        # `LiveStartRequest` validation can happen без CUDA cost
        self._yolo: YoloPoseRunner | None = None
        self._item_runner: YoloItemRunner | None = None
        self._tracker: ByteTrackWrapper | None = None
        self._scorer: BehaviorScorer | None = None
        self._last_cleanup_frame = 0

        # Item detection runs less frequently than pose (items don't move
        # much — running per-frame doubles CPU cost for little gain).
        # Default: every 5 inference cycles ≈ 1 FPS on a 5 FPS pose stream.
        self._item_every_n = 5
        self._inference_count = 0
        self._cached_items: list[Item] = []

        # Config delivered by poller BEFORE scorer initializes — buffer and
        # apply once scorer is live (fixes config-poller race condition).
        self._pending_weights: dict[str, float] | None = None
        self._pending_thresholds: tuple[float, float, float | None] | None = None

        # Latest annotated frame for /v1/live/snapshot/{cam} debug endpoint.
        # Stored as JPEG bytes to avoid GIL contention; updated under lock.
        self._snapshot_lock = threading.Lock()
        self._latest_snapshot_jpeg: bytes | None = None
        self._latest_snapshot_ts: float = 0.0

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

    def apply_weights(self, weights: dict[str, float]) -> None:
        """Hot-update behavior scorer weights from poller.

        If the scorer hasn't initialized yet (worker thread still warming up
        YOLO), buffer the values and apply on init. The poller may not re-fire
        the same config again, so dropping here would mean we run on hardcoded
        defaults until DB changes.
        """
        if self._scorer is not None:
            self._scorer.update_weights(weights)
        else:
            self._pending_weights = dict(weights)

    def apply_thresholds(
        self, green_max: float, yellow_max: float, high_max: float | None = None
    ) -> None:
        """Hot-update level band thresholds from poller. Buffers if scorer not ready."""
        if self._scorer is not None:
            self._scorer.update_thresholds(green_max, yellow_max, high_max)
        else:
            self._pending_thresholds = (green_max, yellow_max, high_max)

    def latest_snapshot(self) -> tuple[bytes, float] | None:
        """Return (jpeg_bytes, unix_ts) of most recent annotated frame, or None."""
        with self._snapshot_lock:
            if self._latest_snapshot_jpeg is None:
                return None
            return self._latest_snapshot_jpeg, self._latest_snapshot_ts

    # === Worker loop ===

    def _run(self) -> None:
        # Lazy heavy init inside the worker thread (avoid blocking the API call)
        try:
            self._yolo = YoloPoseRunner(conf=self.yolo_conf)
            self._item_runner = YoloItemRunner(conf=0.40)
            # ByteTrack frame_rate hint = effective post-skip FPS target
            self._tracker = ByteTrackWrapper(frame_rate=max(1, 30 // self.frame_skip))
            self._scorer = BehaviorScorer()
            # Apply any config the poller delivered while we were initializing.
            if self._pending_weights is not None:
                self._scorer.update_weights(self._pending_weights)
                self._pending_weights = None
            if self._pending_thresholds is not None:
                self._scorer.update_thresholds(*self._pending_thresholds)
                self._pending_thresholds = None
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
                    self._process_frame(frame.astype(np.uint8, copy=False))
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

    def _process_frame(self, frame_bgr: NDArray[np.uint8]) -> None:
        assert self._yolo is not None
        assert self._item_runner is not None
        assert self._tracker is not None
        assert self._scorer is not None

        h, w = frame_bgr.shape[:2]
        detections = self._yolo.detect_persons(frame_bgr)
        tracked = self._tracker.update(detections)
        self._inference_times.append(time.monotonic())
        self._detections_total += len(detections)

        # Item detection on every Nth inference cycle (items don't move much).
        self._inference_count += 1
        if self._inference_count % self._item_every_n == 0:
            try:
                self._cached_items = self._item_runner.detect_items(frame_bgr)
            except Exception:  # noqa: BLE001
                log.exception("camera.item_det_failed", camera_id=self.camera_id)

        # v2 behavior engine per tracked person → risk_pct + level + state + color
        tracks_payload: list[TrackPayload] = []
        for t in tracked:
            person_h = max(1.0, t.box[3] - t.box[1])
            result = self._scorer.score(
                t.tracker_id,
                t.keypoints,
                person_h,
                items=self._cached_items,
            )
            risk_pct = result.risk_pct  # absolute 0-100 (ADR-0024)

            # Cross-camera re-ID + score accumulation (ADR-0022/0023). Link this
            # person to a store-global id and add this frame's positive increment
            # to their store-wide total, so suspicion built across cameras carries.
            store_person_id: int | None = None
            store_risk_pct: float | None = None
            if self._registry is not None and self._embedder is not None:
                emb = self._embedder.embed(frame_bgr, t.box)
                if emb is not None:
                    store_person_id = self._registry.match_or_create(emb, self.camera_id)
                    delta = max(0.0, result.raw_score - self._prev_raw.get(t.tracker_id, 0.0))
                    store_total = self._registry.add_score(store_person_id, delta)
                    store_risk_pct = self._scorer.risk_pct(store_total)
            self._prev_raw[t.tracker_id] = result.raw_score

            tracks_payload.append(
                TrackPayload(
                    person_id=t.tracker_id,
                    box=t.box,
                    det_confidence=t.score,
                    risk_pct=risk_pct,
                    color=result.color,
                    level=result.level,
                    state=result.state.name,
                    sequences=result.sequences,
                    store_person_id=store_person_id,
                    store_risk_pct=store_risk_pct,
                ),
            )

        # Periodic stale-track cleanup (~once per second @ 5 FPS)
        if self._frames_total - self._last_cleanup_frame > 30:
            self._scorer.cleanup_stale()
            self._last_cleanup_frame = self._frames_total

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

        # Update annotated snapshot for /v1/live/snapshot/{cam} debug viewer
        self._update_snapshot(frame_bgr, tracked, tracks_payload)

    def _update_snapshot(
        self,
        frame_bgr: NDArray[np.uint8],
        tracked: list[TrackedDetection],
        tracks_payload: list[TrackPayload],
    ) -> None:
        """Draw bboxes onto a copy of the frame and JPEG-encode for the debug endpoint."""
        annotated = frame_bgr.copy()
        # tracked and tracks_payload are aligned 1:1 by build order in _process_frame
        for t, p in zip(tracked, tracks_payload, strict=False):
            x1, y1, x2, y2 = (int(v) for v in t.box)
            # Color in BGR (cv2) — green / yellow / red per risk band
            if p.color == "red":
                bgr = (0, 0, 255)
            elif p.color == "yellow":
                bgr = (0, 230, 230)
            else:
                bgr = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 3)
            # Normalized 0-100 risk (ADR-0022)
            label = f"#{t.tracker_id}  Risk: {p.risk_pct:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Header overlay: cam + frame # + FPS + det count
        header = (
            f"{self.camera_id}  frame={self._frames_total}  "
            f"FPS={self.fps_inference:.1f}  persons={len(tracked)}"
        )
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            header,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._snapshot_lock:
            self._latest_snapshot_jpeg = buf.tobytes()
            self._latest_snapshot_ts = time.time()


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
