"""Held-item overlay payload: Mongolian labels + the wrist-over-item 'held' flag
the browser live overlay uses to box + name what a shopper is carrying."""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.camera_worker import _build_items_payload, item_label_mn
from sentry_ai.live_worker.tracker import TrackedDetection
from sentry_ai.live_worker.yolo_det import Item


def _person_with_wrist(x: float, y: float) -> TrackedDetection:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[10] = (x, y, 0.9)  # right wrist, confident
    return TrackedDetection(box=(250.0, 100.0, 350.0, 400.0), score=0.9, tracker_id=1, keypoints=kp)


def test_item_label_mn_maps_and_falls_back() -> None:
    assert item_label_mn("bottle") == "лонх"
    assert item_label_mn("Handbag") == "гар цүнх"  # case-insensitive
    assert item_label_mn("widget-9000") == "widget-9000"  # unmapped → raw


def test_held_flag_true_when_wrist_over_item() -> None:
    person = _person_with_wrist(310.0, 300.0)
    items = [Item("bottle", (305.0, 300.0, 325.0, 320.0), 0.8)]
    out = _build_items_payload(items, [person])
    assert len(out) == 1
    assert out[0].held is True
    assert out[0].label_mn == "лонх"
    assert out[0].label == "bottle"


def test_held_flag_false_when_no_wrist_near() -> None:
    person = _person_with_wrist(310.0, 300.0)
    items = [Item("bottle", (600.0, 50.0, 620.0, 70.0), 0.8)]  # far from the wrist
    out = _build_items_payload(items, [person])
    assert len(out) == 1
    assert out[0].held is False


def test_empty_items_yield_empty_payload() -> None:
    assert _build_items_payload([], [_person_with_wrist(1.0, 1.0)]) == []
