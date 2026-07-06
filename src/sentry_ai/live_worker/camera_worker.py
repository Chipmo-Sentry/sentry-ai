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
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from sentry_ai.live_worker import breach_pusher
from sentry_ai.live_worker.behavior import BehaviorScorer, ScoreResult
from sentry_ai.live_worker.emitter import MetadataEmitter
from sentry_ai.live_worker.reid import Embedder, StorePersonRegistry
from sentry_ai.live_worker.schemas import FrameMetadata, ItemPayload, TrackPayload
from sentry_ai.live_worker.tracker import ByteTrackWrapper, TrackedDetection
from sentry_ai.live_worker.yolo_det import Item, YoloItemRunner
from sentry_ai.live_worker.yolo_runner import YoloPoseRunner
from sentry_ai.live_worker.zones import compile_zones, zones_at
from sentry_ai.logging_setup import get_logger
from sentry_ai.runtime_config import get_detection
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.live_worker.camera")

# Rolling window for FPS smoothing
_FPS_WINDOW_SEC = 5.0

# How often to emit the per-camera YOLO filter heartbeat log (seconds).
# Proves the worker is consuming frames, the frame_skip gate is dropping the
# expected fraction, and YOLO is actually detecting persons.
_STATS_LOG_SEC = 5.0

# Re-ID quality gate: only skip people too small/far to identify reliably.
# We deliberately DON'T gate on keypoint visibility — torso cropping already
# falls back to a body-proportion estimate when shoulders/hips aren't visible,
# so gating on keypoints wrongly skipped people on overhead/fisheye cameras
# (hips hidden behind desks) and suppressed cross-camera re-ID entirely.
_REID_MIN_BOX_FRAC = 0.08
_REID_MIN_BOX_PX = 64.0

# === Debug-snapshot overlay (segmentation mask + trail + wrist→item link) ===
# COCO-17 keypoint indices (mirror behavior.py) used to draw the pose-polygon
# body "mask" (option A: no extra seg model — approximate the silhouette from
# the pose the worker already produces).
_KP_NOSE = 0
_KP_L_SHO, _KP_R_SHO = 5, 6
_KP_L_WRI, _KP_R_WRI = 9, 10
_KP_L_HIP, _KP_R_HIP = 11, 12
# Limb chains drawn as thick rounded strokes; torso quad filled for body bulk.
_LIMB_CHAINS: tuple[tuple[int, ...], ...] = (
    (5, 7, 9),  # left shoulder → elbow → wrist
    (6, 8, 10),  # right shoulder → elbow → wrist
    (11, 13, 15),  # left hip → knee → ankle
    (12, 14, 16),  # right hip → knee → ankle
)
# Order makes a non-self-intersecting quad: L-shoulder → R-shoulder → R-hip → L-hip.
_TORSO_QUAD = (_KP_L_SHO, _KP_R_SHO, _KP_R_HIP, _KP_L_HIP)
_MIN_KP_CONF_DRAW = 0.30  # joints below this confidence aren't drawn
_TRAIL_MAXLEN = 32  # foot points kept per track for the trajectory trail
_TRAIL_TTL_SEC = 3.0  # drop a track's trail this long after it vanishes
_MASK_ALPHA = 0.40  # translucency of the pose-polygon body fill
_ITEM_LINK_BGR = (0, 170, 255)  # amber wrist→item link (BGR)
_ITEM_REACH_FRAC = 0.35  # wrist→item "holding" distance as a fraction of person height

# Mongolian display names for detected items (mirrors agent-pc edge/overlay.py).
# The live overlay shows WHAT is in a shopper's hand; computed node-side so the
# browser just renders text. COCO item labels here (the cloud detector is COCO);
# unmapped labels fall back to the raw English label.
ITEM_LABELS_MN: dict[str, str] = {
    "backpack": "үүргэвч",
    "handbag": "гар цүнх",
    "suitcase": "чемодан",
    "bottle": "лонх",
    "cup": "аяга",
    "laptop": "зөөврийн компьютер",
    "cell phone": "гар утас",
    "book": "ном",
    "box": "хайрцаг",
    "can": "лааз",
}


def item_label_mn(label: str) -> str:
    """Mongolian display name for a detected item, or the raw label if unmapped."""
    return ITEM_LABELS_MN.get(label.lower(), label)


def _build_items_payload(items: list[Item], tracked: list[TrackedDetection]) -> list[ItemPayload]:
    """Detected items → overlay payload, flagging each one a wrist is over as
    `held` (the 'in hand' geometry), so the browser can box + name what's carried."""
    out: list[ItemPayload] = []
    for it in items:
        ix1, iy1, ix2, iy2 = it.box
        held = False
        for t in tracked:
            reach = max(1.0, t.box[3] - t.box[1]) * _ITEM_REACH_FRAC
            for widx in (_KP_L_WRI, _KP_R_WRI):
                w = _kp_point(t.keypoints, widx)
                if w is None:
                    continue
                nx = min(max(float(w[0]), ix1), ix2)
                ny = min(max(float(w[1]), iy1), iy2)
                if ((w[0] - nx) ** 2 + (w[1] - ny) ** 2) ** 0.5 <= reach:
                    held = True
                    break
            if held:
                break
        out.append(
            ItemPayload(
                box=it.box,
                label=it.label,
                label_mn=item_label_mn(it.label),
                confidence=float(it.score),
                held=held,
            )
        )
    return out

# Фаз 0 pose-history capture (training data): keep ~this many analyzed frames per
# track (~30 s at a ~5 fps analyzed rate), prune a track after this idle TTL.
# Keypoints are rounded to 1 decimal to bound the JSON size of an alert.
_POSE_HISTORY_MAXLEN = 150
_POSE_HISTORY_TTL_SEC = 60.0


def _risk_bgr(color: str) -> tuple[int, int, int]:
    """Risk band → BGR (cv2). Matches the existing box colours."""
    if color == "red":
        return (0, 0, 255)
    if color == "yellow":
        return (0, 230, 230)
    return (0, 255, 0)


def _kp_point(kp: NDArray[np.float32] | None, idx: int) -> tuple[int, int] | None:
    """(x, y) pixel of keypoint `idx`, or None if unset / below draw-confidence."""
    if kp is None or idx >= kp.shape[0]:
        return None
    row = kp[idx]
    x, y = float(row[0]), float(row[1])
    if not (x > 1.0 and y > 1.0):  # (0,0)-ish → keypoint not detected
        return None
    if kp.shape[1] >= 3 and float(row[2]) < _MIN_KP_CONF_DRAW:
        return None
    return int(x), int(y)


def _draw_body_fill(
    overlay: NDArray[np.uint8],
    kp: NDArray[np.float32] | None,
    bgr: tuple[int, int, int],
    person_h: float,
) -> bool:
    """Approximate a body silhouette from pose keypoints (option A). Draws onto
    `overlay` (later alpha-blended). Returns True if anything was drawn."""
    if kp is None:
        return False
    th = max(4, int(person_h * 0.07))
    drew = False
    quad = [p for p in (_kp_point(kp, i) for i in _TORSO_QUAD) if p is not None]
    if len(quad) >= 3:
        cv2.fillPoly(overlay, [np.array(quad, dtype=np.int32)], bgr)
        drew = True
    for chain in _LIMB_CHAINS:
        prev = None
        for idx in chain:
            cur = _kp_point(kp, idx)
            if cur is not None:
                cv2.circle(overlay, cur, max(3, th // 2), bgr, -1, cv2.LINE_AA)
                if prev is not None:
                    cv2.line(overlay, prev, cur, bgr, th, cv2.LINE_AA)
                    drew = True
            prev = cur
    nose = _kp_point(kp, _KP_NOSE)
    if nose is not None:
        cv2.circle(overlay, nose, max(6, int(person_h * 0.06)), bgr, -1, cv2.LINE_AA)
        drew = True
    return drew


def _draw_wrist_item_links(
    img: NDArray[np.uint8],
    kp: NDArray[np.float32] | None,
    items: list[Item],
    person_h: float,
) -> None:
    """Link a wrist to any nearby item box — visualises the 'holding' geometry."""
    if kp is None or not items:
        return
    reach = person_h * 0.35
    for widx in (_KP_L_WRI, _KP_R_WRI):
        w = _kp_point(kp, widx)
        if w is None:
            continue
        for it in items:
            ix1, iy1, ix2, iy2 = it.box
            nx = min(max(w[0], ix1), ix2)
            ny = min(max(w[1], iy1), iy2)
            if ((w[0] - nx) ** 2 + (w[1] - ny) ** 2) ** 0.5 <= reach:
                cv2.rectangle(img, (int(ix1), int(iy1)), (int(ix2), int(iy2)), _ITEM_LINK_BGR, 2)
                cv2.line(
                    img,
                    w,
                    (int((ix1 + ix2) / 2), int((iy1 + iy2) / 2)),
                    _ITEM_LINK_BGR,
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(img, w, 4, _ITEM_LINK_BGR, -1, cv2.LINE_AA)


def _reid_quality_ok(box: tuple[float, float, float, float], frame_h: int) -> bool:
    """True if this detection is large enough to produce a usable re-ID vector."""
    box_h = box[3] - box[1]
    return box_h >= max(_REID_MIN_BOX_PX, _REID_MIN_BOX_FRAC * frame_h)


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
        alert_threshold_pct: float | None = None,
        zones: list[dict[str, object]] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        # docs/29 P1c — per-camera detection zones. `zones` keeps the raw list for
        # the manager's restart-on-change check; `_compiled_zones` is the
        # {type: [polygon]} form the per-frame point-in-zone test uses.
        self.zones = zones or None
        self._compiled_zones = compile_zones(zones)
        # Constructor frame_skip is the FALLBACK; the live value comes from the
        # central per-node config (runtime_config) and is read fresh each frame so
        # an operator edit in superadmin takes effect without restarting the worker.
        self.frame_skip = max(1, frame_skip)
        # The frame_skip actually applied on the latest read (effective value, for
        # the yolo_stats log + heartbeat). Starts at the constructor fallback.
        self._active_frame_skip = self.frame_skip
        # Per-camera scan floor (0-100) from the backend (risk_threshold). None →
        # fall back to the node-global default. Since the VLM-primary pivot this
        # gates which people are ELIGIBLE for the periodic VLM scan (docs/33 P0-4)
        # — it no longer fires alerts directly (the VLM decides those).
        self.alert_threshold_pct = alert_threshold_pct
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
        self._frames_total = 0  # frames READ off RTSP (pre frame_skip)
        self._frames_processed = 0  # frames that PASSED frame_skip → YOLO ran
        self._detections_total = 0
        self._persons_this_frame = 0  # persons in the latest analyzed frame (for heartbeat)
        self._last_stats_log = 0.0  # monotonic ts of last yolo_stats log
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

        # Node-push live alerts: per-track sustain timer + cooldown. On a
        # sustained breach we hand off to breach_pusher (cut + VLM + POST to the
        # backend) on its own worker — this frame loop never blocks on it.
        _s = get_settings()
        # VLM-primary scan cadence: every N seconds, hand the most-suspicious
        # PRESENT person to the VLM (cut + verify + POST) and let the AI decide —
        # the behaviour engine no longer GATES this, it just prioritises who to
        # scan. breach_pusher drops ticks while the GPU is busy.
        self._scan_interval = max(0.5, _s.live_scan_interval_sec)
        self._last_scan = 0.0
        self._alert_threshold_pct = (
            alert_threshold_pct if alert_threshold_pct is not None else _s.live_alert_threshold_pct
        )
        # docs/33 P0-4 — scan cost controls: the scan only considers people at or
        # above _alert_threshold_pct (re-wiring the previously-dead per-camera
        # risk_threshold as the scan FLOOR), and the same person is not re-scanned
        # within this cooldown. GPU spend now scales with suspicion, not foot
        # traffic (previously: risk-0 shoppers were VLM-scanned every 3 s, 24/7).
        self._scan_person_cooldown = max(0.0, _s.live_scan_person_cooldown_sec)
        self._scan_last_by_tid: dict[int, float] = {}

        # Config delivered by poller BEFORE scorer initializes — buffer and
        # apply once scorer is live (fixes config-poller race condition).
        self._pending_weights: dict[str, float] | None = None
        self._pending_thresholds: tuple[float, float, float | None] | None = None
        self._pending_params: tuple[dict[str, float], dict[str, dict[str, float]]] | None = None

        # Latest annotated frame for /v1/live/snapshot/{cam} debug endpoint.
        # Stored as JPEG bytes to avoid GIL contention; updated under lock.
        self._snapshot_lock = threading.Lock()
        self._latest_snapshot_jpeg: bytes | None = None
        self._latest_snapshot_ts: float = 0.0
        # Per-track foot-point history for the trajectory-trail overlay:
        # tracker_id → (recent points, last-updated wall-clock). Pruned by TTL.
        self._trail_hist: dict[int, tuple[deque[tuple[int, int]], float]] = {}
        # Фаз 0 (ADR-0030): per-track rolling pose history → at breach we attach the
        # breaching person's skeleton trajectory to the alert as training data.
        # tracker_id → (deque of {frame_idx, ts_ms, keypoints, box}, last-update ts).
        self._pose_history: dict[int, tuple[deque[dict[str, Any]], float]] = {}

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
    def effective_frame_skip(self) -> int:
        """The frame_skip actually applied on the latest read (live central value
        or the constructor fallback) — surfaced in /v1/live/status for the heartbeat."""
        return self._active_frame_skip

    @property
    def persons_this_frame(self) -> int:
        """Persons in the latest analyzed frame — surfaced for the heartbeat so the
        YOLO stage shows real per-camera numbers."""
        return self._persons_this_frame

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

    def apply_params(
        self,
        engine: dict[str, float],
        detector: dict[str, dict[str, float]],
    ) -> None:
        """Hot-update engine + per-detector tuning params from poller.
        Buffers if the scorer hasn't initialized yet (same race as weights)."""
        if self._scorer is not None:
            self._scorer.update_params(engine, detector)
        else:
            self._pending_params = (engine, detector)

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
            if self._pending_params is not None:
                self._scorer.update_params(*self._pending_params)
                self._pending_params = None
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
                    # Recorded so per-camera health (heartbeat `cameras`) reports
                    # this camera as "error" instead of silently showing 0 FPS.
                    self._last_error = "RTSP read failed"
                    log.warning("camera.read_failed", camera_id=self.camera_id)
                    cap.release()
                    cap = None
                    continue

                self._frames_total += 1
                self._capture_times.append(time.monotonic())

                # Effective frame_skip from the central per-node config (live), so a
                # superadmin edit changes the analyzed FPS without a worker restart.
                self._active_frame_skip = self._effective_frame_skip()
                if self._frames_total % self._active_frame_skip != 0:
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
        # Successful (re)connect clears the sticky error so health recovers.
        self._last_error = None
        log.info("camera.open_ok", camera_id=self.camera_id)
        return cap

    def _effective_frame_skip(self) -> int:
        """Live frame_skip: the central per-node config wins, else the constructor
        fallback. Read fresh every captured frame so an edit applies immediately."""
        cfg = get_detection()
        if cfg is not None and cfg.frame_skip is not None:
            return max(1, cfg.frame_skip)
        return self.frame_skip

    def _sync_detection_config(self) -> None:
        """Apply the central per-node YOLO/scan tuning to this running worker.
        Cheap (a few attribute writes) and called once per analyzed frame; the
        YOLO runners read `conf` on every predict, so updating it here takes effect
        on the NEXT inference with no model reload. Unset fields keep their own
        constructor/settings default."""
        cfg = get_detection()
        if cfg is None:
            return
        if cfg.person_conf is not None and self._yolo is not None:
            self._yolo.conf = cfg.person_conf
        if cfg.item_conf is not None and self._item_runner is not None:
            self._item_runner.conf = cfg.item_conf
        if cfg.item_every_n is not None:
            self._item_every_n = max(1, cfg.item_every_n)
        if cfg.scan_interval_sec is not None:
            self._scan_interval = max(0.5, cfg.scan_interval_sec)

    def _process_frame(self, frame_bgr: NDArray[np.uint8]) -> None:
        assert self._yolo is not None
        assert self._item_runner is not None
        assert self._tracker is not None
        assert self._scorer is not None

        # Pull the latest central per-node config onto this worker (conf / item
        # cadence / scan interval). frame_skip is applied in the read loop.
        self._sync_detection_config()

        h, w = frame_bgr.shape[:2]
        detections = self._yolo.detect_persons(frame_bgr)
        tracked = self._tracker.update(detections)
        now = time.monotonic()
        self._inference_times.append(now)
        self._frames_processed += 1
        self._detections_total += len(detections)
        self._persons_this_frame = len(detections)
        self._capture_pose(tracked)  # Фаз 0: roll the per-track skeleton history

        # Throttled YOLO-filter heartbeat. captured vs processed ≈ frame_skip
        # confirms the skip gate; persons>0 when someone is in view confirms the
        # detector + conf threshold are live. Grep `camera.yolo_stats` in logs.
        if now - self._last_stats_log >= _STATS_LOG_SEC:
            self._last_stats_log = now
            log.info(
                "camera.yolo_stats",
                camera_id=self.camera_id,
                frames_captured=self._frames_total,
                frames_processed=self._frames_processed,
                frame_skip=self._active_frame_skip,
                persons_this_frame=len(detections),
                tracks_this_frame=len(tracked),
                detections_total=self._detections_total,
                yolo_conf=round(self._yolo.conf, 2),
                fps_capture=round(self.fps_capture, 1),
                fps_inference=round(self.fps_inference, 1),
            )

        # Item detection on every Nth inference cycle (items don't move much).
        self._inference_count += 1
        if self._inference_count % self._item_every_n == 0:
            try:
                self._cached_items = self._item_runner.detect_items(frame_bgr)
            except Exception:  # noqa: BLE001
                log.exception("camera.item_det_failed", camera_id=self.camera_id)

        # v2 behavior engine per tracked person → risk_pct + level + state + color
        tracks_payload: list[TrackPayload] = []
        scan_candidates: list[tuple[float, int, ScoreResult]] = []
        for t in tracked:
            person_h = max(1.0, t.box[3] - t.box[1])
            # docs/29 P1c — which zone type(s) the person's FOOT point is inside.
            # Normalize against THIS frame's w,h (risk #3: real frame, not nominal
            # res). Skip the test entirely when the camera has no zones.
            in_zones: set[str] | None = None
            if self._compiled_zones:
                foot_x = (t.box[0] + t.box[2]) / 2.0
                in_zones = zones_at(foot_x / max(1, w), t.box[3] / max(1, h), self._compiled_zones)
            result = self._scorer.score(
                t.tracker_id,
                t.keypoints,
                person_h,
                items=self._cached_items,
                in_zones=in_zones,
            )
            risk_pct = result.risk_pct  # absolute 0-100 (ADR-0024)

            # Cross-camera re-ID + score accumulation (ADR-0022/0023). Link this
            # person to a store-global id and add this frame's positive increment
            # to their store-wide total, so suspicion built across cameras carries.
            store_person_id: int | None = None
            store_risk_pct: float | None = None
            if (
                self._registry is not None
                and self._embedder is not None
                and _reid_quality_ok(t.box, h)
            ):
                # Crop to the torso (via keypoints) and embed — see reid.py.
                emb = self._embedder.embed(frame_bgr, t.box, t.keypoints)
                if emb is not None:
                    store_person_id = self._registry.match_or_create(emb, self.camera_id)
                    delta = max(0.0, result.raw_score - self._prev_raw.get(t.tracker_id, 0.0))
                    store_total = self._registry.add_score(store_person_id, delta)
                    store_risk_pct = self._scorer.risk_pct(store_total)
                    # Advance the baseline ONLY when the increment was actually banked,
                    # so risk built up during low-quality/skipped frames isn't lost — it
                    # carries to the next frame clear enough to re-identify the person.
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
                    behaviors=result.behaviors,
                    behavior_scores={k: round(v, 1) for k, v in result.behavior_scores.items()},
                    behavior_offsets=result.behavior_offsets,
                    reasons=result.reasons,
                    episode_started_ms=(
                        int(result.episode_started_at * 1000)
                        if result.episode_started_at is not None
                        else None
                    ),
                    store_person_id=store_person_id,
                    store_risk_pct=store_risk_pct,
                ),
            )

            # VLM-primary: collect scan CANDIDATES — people at or above the scan
            # floor (docs/33 P0-4: the per-camera risk_threshold, previously a
            # dead knob). The periodic scan below picks the most suspicious
            # eligible one (per-person cooldown applies); the AI decides — the
            # behaviour engine no longer GATES alerts, it only prioritises.
            if risk_pct >= self._alert_threshold_pct:
                scan_candidates.append((risk_pct, t.tracker_id, result))

        # Periodic stale-track cleanup (~once per second @ 5 FPS)
        if self._frames_total - self._last_cleanup_frame > 30:
            self._scorer.cleanup_stale()
            self._last_cleanup_frame = self._frames_total
            # Prune the per-person scan cooldown map with the same staleness rule
            # the old breach state used, so it can't grow with ByteTrack's
            # monotonically-increasing ids.
            stale_scans = [tid for tid, ts in self._scan_last_by_tid.items() if now - ts > 300.0]
            for tid in stale_scans:
                del self._scan_last_by_tid[tid]

        # VLM-primary periodic scan (docs/33 P0-4): every scan_interval, hand the
        # most-suspicious ELIGIBLE person to the VLM — at/above the scan floor
        # (collected above) and not scanned within the per-person cooldown. A tick
        # with no eligible candidate submits nothing (idle store = idle GPU).
        # breach_pusher still drops ticks while the GPU is busy.
        if scan_candidates and now - self._last_scan >= self._scan_interval:
            scan_candidates.sort(key=lambda c: c[0], reverse=True)
            for risk, tid, res in scan_candidates:
                last = self._scan_last_by_tid.get(tid)
                if last is not None and now - last < self._scan_person_cooldown:
                    continue
                self._last_scan = now
                self._scan_last_by_tid[tid] = now
                self._submit_breach(tid, risk, res)
                break

        payload = FrameMetadata(
            camera_id=self.camera_id,
            frame_id=self._frames_total,
            ts_ms=int(time.time() * 1000),
            width=w,
            height=h,
            fps_inference=self.fps_inference,
            tracks=tracks_payload,
            items=_build_items_payload(self._cached_items, tracked),
        )
        self._emitter.enqueue(payload)

        # Update annotated snapshot for /v1/live/snapshot/{cam} debug viewer
        self._update_snapshot(frame_bgr, tracked, tracks_payload)

    def _submit_breach(self, tracker_id: int, peak_risk_pct: float, result: ScoreResult) -> None:
        # Per-FIRE breakdown (each +score increment is its own row). Fall back to
        # the per-criterion summary for safety / older episodes.
        detail = result.behavior_events or [
            {
                "key": k,
                "offset_sec": round(float(result.behavior_offsets.get(k, 0.0)), 1),
                "score": round(float(result.behavior_scores.get(k, 0.0)), 1),
            }
            for k in result.behaviors
        ]
        episode_ms = (
            int(result.episode_started_at * 1000) if result.episode_started_at is not None else None
        )
        log.info(
            "camera.breach",
            camera_id=self.camera_id,
            person_id=tracker_id,
            peak_risk_pct=round(peak_risk_pct, 1),
            behaviors=result.behaviors,
        )
        breach_pusher.submit_breach(
            mediamtx_path=self.camera_id,
            person_id=tracker_id,
            peak_risk_pct=peak_risk_pct,
            behaviors=list(result.behaviors),
            sequences=list(result.sequences),
            behavior_detail=detail,
            episode_started_ms=episode_ms,
            breach_ts_ms=int(time.time() * 1000),
            pose_sequence=self._build_pose_sequence(tracker_id) or None,
        )

    def _capture_pose(self, tracked: list[TrackedDetection]) -> None:
        """Фаз 0: roll each present person's pose into the per-track history (skeleton
        only — no pixels). At breach, the breaching track's window becomes the
        alert's training data. Keypoints rounded to bound the stored JSON size."""
        now = time.time()
        for t in tracked:
            if t.keypoints is None:
                continue
            entry = self._pose_history.get(t.tracker_id)
            dq = entry[0] if entry is not None else deque(maxlen=_POSE_HISTORY_MAXLEN)
            dq.append(
                {
                    "frame_idx": self._frames_total,
                    "ts_ms": int(now * 1000),
                    "keypoints": np.round(t.keypoints, 1).tolist(),
                    "box": [round(float(v), 1) for v in t.box],
                }
            )
            self._pose_history[t.tracker_id] = (dq, now)
        # Prune tracks idle past the TTL so the dict can't grow without bound.
        for tid in [
            k for k, (_, ts) in self._pose_history.items() if now - ts > _POSE_HISTORY_TTL_SEC
        ]:
            del self._pose_history[tid]

    def _build_pose_sequence(self, tracker_id: int) -> list[dict[str, Any]]:
        """The breaching track's recent skeleton trajectory (oldest→newest), or []."""
        entry = self._pose_history.get(tracker_id)
        return list(entry[0]) if entry is not None else []

    def _update_snapshot(
        self,
        frame_bgr: NDArray[np.uint8],
        tracked: list[TrackedDetection],
        tracks_payload: list[TrackPayload],
    ) -> None:
        """Annotate the frame for the debug endpoint: pose-polygon body mask,
        trajectory trail, wrist→item link, and the risk/behavior box."""
        annotated = frame_bgr.copy()
        items = self._cached_items
        now = time.time()

        # --- Update + prune per-track trajectory history (foot point) ---
        for t in tracked:
            foot = (int((t.box[0] + t.box[2]) / 2), int(t.box[3]))
            hist_e = self._trail_hist.get(t.tracker_id)
            dq = hist_e[0] if hist_e is not None else deque(maxlen=_TRAIL_MAXLEN)
            dq.append(foot)
            self._trail_hist[t.tracker_id] = (dq, now)
        for tid in [k for k, (_, ts) in self._trail_hist.items() if now - ts > _TRAIL_TTL_SEC]:
            del self._trail_hist[tid]

        # --- Layer 1: translucent pose-polygon body "mask" (option A) ---
        overlay = annotated.copy()
        drew_mask = False
        for t, p in zip(tracked, tracks_payload, strict=False):
            person_h = max(1.0, t.box[3] - t.box[1])
            if _draw_body_fill(overlay, t.keypoints, _risk_bgr(p.color), person_h):
                drew_mask = True
        if drew_mask:
            cv2.addWeighted(overlay, _MASK_ALPHA, annotated, 1 - _MASK_ALPHA, 0, annotated)

        # --- Layers 2-4: trail, wrist→item link, risk box + labels (opaque) ---
        # tracked and tracks_payload are aligned 1:1 by build order in _process_frame
        for t, p in zip(tracked, tracks_payload, strict=False):
            x1, y1, x2, y2 = (int(v) for v in t.box)
            bgr = _risk_bgr(p.color)
            person_h = max(1.0, t.box[3] - t.box[1])

            # Trajectory trail — foot path over recent frames.
            hist = self._trail_hist.get(t.tracker_id)
            if hist is not None and len(hist[0]) >= 2:
                pts = np.array(hist[0], dtype=np.int32)
                cv2.polylines(annotated, [pts], False, bgr, 2, cv2.LINE_AA)
                cv2.circle(annotated, (int(pts[-1][0]), int(pts[-1][1])), 4, bgr, -1, cv2.LINE_AA)

            # Wrist → item link (amber) when a wrist is near a detected item.
            _draw_wrist_item_links(annotated, t.keypoints, items, person_h)

            # Risk box + #id Risk% label (ADR-0024 absolute 0-100).
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 3)
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
            # Active criteria keys under the box (ASCII — cv2 can't render Cyrillic).
            if p.behaviors:
                shown = p.behaviors[:3]
                more = len(p.behaviors) - len(shown)
                crit_line = ", ".join(shown) + (f" +{more}" if more > 0 else "")
                cv2.putText(
                    annotated,
                    crit_line,
                    (x1 + 2, min(annotated.shape[0] - 4, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    bgr,
                    1,
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
