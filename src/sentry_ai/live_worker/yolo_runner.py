"""YOLO11-pose inference wrapper.

Lazy-loads the model on first call (downloads weights ~6 MB from Ultralytics
hub if missing). Auto-selects CUDA when available, falls back to CPU.

Returns a list of (box, score) tuples for `person` class only (class id 0
in COCO).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.yolo")

_MODEL_LOCK = threading.Lock()
_MODEL: object | None = None
_DEVICE: str | None = None

# Person class index in COCO80 (YOLO default labels)
PERSON_CLASS = 0


@dataclass(slots=True)
class Detection:
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    score: float                              # 0.0-1.0
    # COCO-17 pose keypoints. Shape (17, 2) = [x, y] per joint.
    # Order: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow,
    # r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle.
    # Values are pixel coords in the input frame; (0, 0) means "not detected".
    # None if model is not pose-capable (defensive default).
    keypoints: NDArray[np.float32] | None = None


def _load_model(weights: str = "yolo11n-pose.pt") -> tuple[object, str]:
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
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
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
        # Trigger lazy load eagerly on construct so first frame inference is fast
        _load_model()

    def detect_persons(self, frame_bgr: NDArray[np.uint8]) -> list[Detection]:
        """Run inference on a single BGR frame, return person detections.

        frame_bgr: (H, W, 3) uint8 array from cv2 (BGR order).
        """
        model, device = _load_model()

        # ultralytics expects HWC BGR ndarray or path; classes=[0] filters to persons
        # verbose=False suppresses per-frame stdout spam
        results = model.predict(  # type: ignore[attr-defined]
            frame_bgr,
            device=device,
            conf=self.conf,
            iou=self.iou,
            classes=[PERSON_CLASS],
            verbose=False,
        )

        if not results:
            return []
        r = results[0]
        # r.boxes: Boxes object with .xyxy (N, 4) and .conf (N,)
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Tensors → CPU numpy
        xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
        conf = boxes.conf.cpu().numpy()       # (N,)

        # Pose keypoints (only present on pose models). r.keypoints.xy: (N, 17, 2)
        kpts_xy: NDArray[np.float32] | None = None
        if r.keypoints is not None and r.keypoints.xy is not None:
            kpts_xy = r.keypoints.xy.cpu().numpy().astype(np.float32)

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


def get_device() -> str | None:
    """Return the device string after first model load, or None if not yet loaded."""
    return _DEVICE


def weights_cached(weights: str = "yolo11n-pose.pt") -> bool:
    """Best-effort check whether weights are already on disk (skips download log)."""
    # ultralytics default cache: ~/.config/Ultralytics/ OR cwd
    candidates = [
        Path.cwd() / weights,
        Path.home() / ".cache" / "ultralytics" / weights,
    ]
    return any(p.exists() for p in candidates)
