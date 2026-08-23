"""YOLO pose inference wrapper (YOLO26-pose default — ADR-0026).

Lazy-loads the model on first call (downloads weights ~6 MB from Ultralytics
hub if missing). Auto-selects CUDA when available, falls back to CPU.

Returns a list of (box, score) tuples for `person` class only (class id 0
in COCO).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.yolo")

_MODEL_LOCK = threading.Lock()
_MODEL: object | None = None
_DEVICE: str | None = None
# docs/33 Sprint C — ONE model is shared by every camera thread, but ultralytics
# predictors keep per-call internal state and are NOT thread-safe: ≥2 cameras
# predicting concurrently can corrupt results or crash. Serialize predict()
# (costs ~nothing: a single GPU serializes the actual compute anyway, while a
# per-worker model would multiply GPU memory).
_PREDICT_LOCK = threading.Lock()

# Person class index in COCO80 (YOLO default labels)
PERSON_CLASS = 0


@dataclass(slots=True)
class Detection:
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    score: float  # 0.0-1.0
    # COCO-17 pose keypoints. Shape (17, 3) = [x, y, conf] per joint (conf is the
    # 3rd column when the model supplies it; (17, 2) otherwise — both accepted by
    # the behavior engine). Order: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder,
    # r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee,
    # r_knee, l_ankle, r_ankle. Pixel coords; (0, 0) means "not detected".
    # None if model is not pose-capable (defensive default).
    keypoints: NDArray[np.float32] | None = None


def _load_model(weights: str = "yolo26s-pose.pt") -> tuple[object, str]:
    """Singleton model load. ultralytics.YOLO caches weights in ~/.cache.

    Returns (model, device_str). device_str = 'cuda:0' or 'cpu'.
    """
    global _MODEL, _DEVICE
    if _MODEL is not None and _DEVICE is not None:
        return _MODEL, _DEVICE

    with _MODEL_LOCK:
        if _MODEL is not None and _DEVICE is not None:
            return _MODEL, _DEVICE

        # Import torch/ultralytics lazily — heavy
        import torch
        from ultralytics import YOLO  # type: ignore[attr-defined]

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("yolo.loading", weights=weights, device=device)
        model = YOLO(weights)
        # Warm up — first inference is significantly slower
        from sentry_ai.settings import get_settings

        # Warm at the CONFIGURED imgsz so the first real frame is fast too.
        sz = get_settings().yolo_imgsz
        dummy = np.zeros((sz, sz, 3), dtype=np.uint8)
        model.predict(dummy, device=device, verbose=False, classes=[PERSON_CLASS])
        log.info("yolo.loaded", device=device)
        _MODEL = model
        _DEVICE = device
        return model, device


class YoloPoseRunner:
    """Thin wrapper around ultralytics YOLO for the live worker."""

    def __init__(self, conf: float = 0.35, iou: float = 0.45) -> None:
        self.conf = conf
        self.iou = iou
        # Trigger lazy load eagerly on construct so first frame inference is fast.
        # Weights configurable (#3) — default yolo26s-pose (ADR-0026).
        from sentry_ai.settings import get_settings

        s = get_settings()
        # Inference tuning: larger imgsz recovers far-away/small persons in wide
        # store scenes; FP16 halves GPU latency (CUDA only — CPU ignores it).
        self.imgsz = s.yolo_imgsz
        self.half = s.yolo_half
        _load_model(s.yolo_pose_weights)

    def detect_persons(self, frame_bgr: NDArray[np.uint8]) -> list[Detection]:
        """Run inference on a single BGR frame, return person detections.

        frame_bgr: (H, W, 3) uint8 array from cv2 (BGR order).

        With yolo_batch on (default), the call routes through the shared
        _PoseBatcher: concurrent cameras' frames ride ONE GPU predict instead
        of queueing on the predict lock — per-call overhead is paid once per
        batch, so multi-camera throughput scales instead of serializing.
        """
        from sentry_ai.settings import get_settings

        if get_settings().yolo_batch:
            batcher = _PoseBatcher.get()
            batcher.set_params(
                conf=self.conf, iou=self.iou, imgsz=self.imgsz, half=self.half
            )
            return _parse_pose_result(batcher.infer(frame_bgr))

        model, device = _load_model()
        # ultralytics expects HWC BGR ndarray or path; classes=[0] filters to persons
        # verbose=False suppresses per-frame stdout spam
        with _PREDICT_LOCK:  # predictors aren't thread-safe across camera threads
            results = model.predict(  # type: ignore[attr-defined]
                frame_bgr,
                device=device,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                half=self.half and device.startswith("cuda"),
                classes=[PERSON_CLASS],
                verbose=False,
            )
        return _parse_pose_result(results[0]) if results else []


def _parse_pose_result(r: object) -> list[Detection]:
    """One ultralytics Results object → person Detections (boxes + keypoints)."""
    if r is None:
        return []
    # r.boxes: Boxes object with .xyxy (N, 4) and .conf (N,)
    boxes = r.boxes  # type: ignore[attr-defined]
    if boxes is None or len(boxes) == 0:
        return []

    # Tensors → CPU numpy
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    conf = boxes.conf.cpu().numpy()  # (N,)

    # Pose keypoints (only present on pose models). r.keypoints.xy: (N, 17, 2),
    # r.keypoints.conf: (N, 17). Stack into (N, 17, 3) [x, y, conf] so the
    # behavior engine can gate low-confidence joints (#1). Fall back to (N,17,2)
    # if confidence isn't available.
    kpts_xy: NDArray[np.float32] | None = None
    keypoints = getattr(r, "keypoints", None)
    if keypoints is not None and keypoints.xy is not None:
        xy = keypoints.xy.cpu().numpy().astype(np.float32)  # (N, 17, 2)
        # NB: must NOT reuse the name `conf` here — that is the box-score
        # array used below for `Detection.score`. Shadowing it broke inference.
        kp_conf = getattr(keypoints, "conf", None)
        if kp_conf is not None:
            c = kp_conf.cpu().numpy().astype(np.float32)[..., None]  # (N, 17, 1)
            kpts_xy = np.concatenate([xy, c], axis=2)  # (N, 17, 3)
        else:
            kpts_xy = xy

    out: list[Detection] = []
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i].tolist()
        kp = kpts_xy[i] if kpts_xy is not None and i < kpts_xy.shape[0] else None
        out.append(
            Detection(
                box=(x1, y1, x2, y2),
                score=float(conf[i]),
                keypoints=kp,
            ),
        )
    return out


class _PoseBatcher:
    """Cross-camera micro-batching for the shared pose model.

    Worker threads submit frames; ONE collector thread drains whatever arrived
    within `window_ms` (or up to `max_batch`) and runs a single batched
    predict — ultralytics letterboxes each list item to imgsz, so mixed frame
    sizes are fine. Results map back positionally. Semantics for the caller are
    identical to a direct predict (blocking, exceptions propagate); the win is
    that N cameras stop paying N× per-call overhead while serialized on the
    predict lock. The collector owns predict exclusively → thread-safe by
    construction.
    """

    _instance: _PoseBatcher | None = None
    _instance_lock = threading.Lock()

    def __init__(self, window_ms: float = 10.0, max_batch: int = 8) -> None:
        import queue

        self.window_s = window_ms / 1000.0
        self.max_batch = max(1, max_batch)
        # Node-global inference params (central config keeps every runner's conf
        # in lockstep anyway); runners refresh them right before each submit.
        self.conf = 0.35
        self.iou = 0.45
        self.imgsz = 640
        self.half = True
        self._q: queue.Queue[tuple[NDArray[np.uint8], Future[object]]] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="yolo-batcher", daemon=True)
        self._thread.start()

    @classmethod
    def get(cls) -> _PoseBatcher:
        with cls._instance_lock:
            if cls._instance is None:
                from sentry_ai.settings import get_settings

                s = get_settings()
                cls._instance = cls(
                    window_ms=s.yolo_batch_window_ms, max_batch=s.yolo_batch_max
                )
            return cls._instance

    def set_params(self, *, conf: float, iou: float, imgsz: int, half: bool) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.half = half

    def infer(self, frame_bgr: NDArray[np.uint8]) -> object:
        """Submit one frame; block until its Results object is ready."""
        fut: Future[object] = Future()
        self._q.put((frame_bgr, fut))
        return fut.result()

    def _loop(self) -> None:
        import queue

        while True:
            frame, fut = self._q.get()  # block for the first frame of a batch
            batch = [(frame, fut)]
            deadline = time.monotonic() + self.window_s
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                model, device = _load_model()
                results = model.predict(  # type: ignore[attr-defined]
                    [b[0] for b in batch],
                    device=device,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    half=self.half and device.startswith("cuda"),
                    classes=[PERSON_CLASS],
                    verbose=False,
                )
            except Exception as e:  # noqa: BLE001 — deliver to callers, keep collecting
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
                continue
            for i, (_, f) in enumerate(batch):
                if not f.done():
                    f.set_result(results[i] if i < len(results) else None)


def get_device() -> str | None:
    """Return the device string after first model load, or None if not yet loaded."""
    return _DEVICE


def weights_cached(weights: str = "yolo26s-pose.pt") -> bool:
    """Best-effort check whether weights are already on disk (skips download log)."""
    # ultralytics default cache: ~/.config/Ultralytics/ OR cwd
    candidates = [
        Path.cwd() / weights,
        Path.home() / ".cache" / "ultralytics" / weights,
    ]
    return any(p.exists() for p in candidates)
