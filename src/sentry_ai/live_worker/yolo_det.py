"""COCO general-object detector — separate from the pose model.

Used by L4.5 to find "expensive items" near person wrists so the
behavior scorer can flip the `holding` state, which in turn activates
the wrist_to_torso and rapid_movement dimensions.

Runs MUCH less often than pose (every Nth pose cycle, configurable)
since items move slowly — caches latest results in the camera worker.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.yolo_det")

# COCO class IDs that matter for retail shrink: small portable items
# someone could conceal. (Person=0 belongs to pose model, excluded here.)
COCO_ITEM_CLASSES: dict[int, str] = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    73: "book",
}

_MODEL_LOCK = threading.Lock()
_MODEL: object | None = None
_DEVICE: str | None = None


@dataclass(slots=True)
class Item:
    label: str
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
    score: float  # 0.0-1.0


def _load_model(weights: str = "yolo11n.pt") -> tuple[object, str]:
    global _MODEL, _DEVICE
    if _MODEL is not None and _DEVICE is not None:
        return _MODEL, _DEVICE

    with _MODEL_LOCK:
        if _MODEL is not None and _DEVICE is not None:
            return _MODEL, _DEVICE

        import torch
        from ultralytics import YOLO  # type: ignore[attr-defined]

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info("yolo_det.loading", weights=weights, device=device)
        model = YOLO(weights)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(
            dummy,
            device=device,
            verbose=False,
            classes=list(COCO_ITEM_CLASSES.keys()),
        )
        log.info("yolo_det.loaded", device=device)
        _MODEL = model
        _DEVICE = device
        return model, device


class YoloItemRunner:
    """Detect retail-shrink-relevant COCO objects in a frame."""

    def __init__(self, conf: float = 0.35, iou: float = 0.5) -> None:
        self.conf = conf
        self.iou = iou
        from sentry_ai.settings import get_settings

        _load_model(get_settings().yolo_item_weights)

    def detect_items(self, frame_bgr: NDArray[np.uint8]) -> list[Item]:
        model, device = _load_model()
        results = model.predict(  # type: ignore[attr-defined]
            frame_bgr,
            device=device,
            conf=self.conf,
            iou=self.iou,
            classes=list(COCO_ITEM_CLASSES.keys()),
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        out: list[Item] = []
        for i in range(xyxy.shape[0]):
            label = COCO_ITEM_CLASSES.get(int(cls[i]))
            if label is None:
                continue
            x1, y1, x2, y2 = xyxy[i].tolist()
            out.append(
                Item(label=label, box=(x1, y1, x2, y2), score=float(conf[i])),
            )
        return out
