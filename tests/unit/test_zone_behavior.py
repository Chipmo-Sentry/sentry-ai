"""Zone-aware behavior criteria (docs/29 P1c): exit_after_concealment +
repeated_shelf_visit. Pure scorer logic — synthetic COCO-17 keypoints, no YOLO."""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.behavior import BehaviorScorer, BehaviorState
from sentry_ai.live_worker.yolo_det import Item

L_EYE, R_EYE = 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
PERSON_H = 200.0


class FakeClock:
    def __init__(self, start: float = 1_781_334_799.64) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _neutral() -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    k[L_SHOULDER] = (90, 100, 1.0)
    k[R_SHOULDER] = (110, 100, 1.0)
    k[L_EYE] = (96, 60, 1.0)
    k[R_EYE] = (104, 60, 1.0)
    k[L_HIP] = (92, 300, 1.0)
    k[R_HIP] = (108, 300, 1.0)
    return k


def _conceal(scorer: BehaviorScorer, tid: int = 1) -> None:
    """Drive a pickup (sets holding) then a wrist-on-hip frame → pocket_interaction
    (a concealment criterion banks into the episode)."""
    k = _neutral()
    k[L_WRIST] = (200, 150, 1.0)
    item = Item(label="cell phone", box=(180, 130, 220, 170), score=0.9)
    scorer.score(tid, k, PERSON_H, items=[item])
    kp = _neutral()
    kp[L_WRIST] = (92, 300, 1.0)  # on the left hip → pocket_interaction
    r = scorer.score(tid, kp, PERSON_H, items=[])
    assert "pocket_interaction" in r.behaviors


# === exit_after_concealment ===


def test_exit_after_concealment_fires_and_alerts() -> None:
    scorer = BehaviorScorer(clock=FakeClock())
    _conceal(scorer)
    r = scorer.score(1, _neutral(), PERSON_H, in_zones={"exit"})
    assert "exit_after_concealment" in r.behaviors
    assert r.state == BehaviorState.ALERT


def test_exit_without_prior_concealment_does_not_fire() -> None:
    scorer = BehaviorScorer(clock=FakeClock())
    # Just standing in the exit zone, no concealment first → no criterion.
    r = scorer.score(1, _neutral(), PERSON_H, in_zones={"exit"})
    assert "exit_after_concealment" not in r.behaviors


def test_exit_after_concealment_banks_once() -> None:
    scorer = BehaviorScorer(clock=FakeClock())
    _conceal(scorer)
    r1 = scorer.score(1, _neutral(), PERSON_H, in_zones={"exit"})
    banked = r1.behavior_scores["exit_after_concealment"]
    r2 = scorer.score(1, _neutral(), PERSON_H, in_zones={"exit"})
    assert r2.behavior_scores["exit_after_concealment"] == banked  # not re-banked


# === repeated_shelf_visit ===


def _enter(scorer: BehaviorScorer, tid: int = 1):  # type: ignore[no-untyped-def]
    return scorer.score(tid, _neutral(), PERSON_H, in_zones={"shelf"})


def _leave(scorer: BehaviorScorer, tid: int = 1):  # type: ignore[no-untyped-def]
    return scorer.score(tid, _neutral(), PERSON_H, in_zones=set())


def test_repeated_shelf_visit_fires_on_third_entry() -> None:
    scorer = BehaviorScorer(clock=FakeClock())
    _enter(scorer)
    _leave(scorer)
    _enter(scorer)
    _leave(scorer)
    r = _enter(scorer)  # 3rd distinct entry → fires (default threshold 3)
    assert "repeated_shelf_visit" in r.behaviors


def test_repeated_shelf_visit_not_before_threshold() -> None:
    scorer = BehaviorScorer(clock=FakeClock())
    _enter(scorer)
    _leave(scorer)
    r = _enter(scorer)  # only 2 entries
    assert "repeated_shelf_visit" not in r.behaviors


def test_staying_in_shelf_is_one_visit() -> None:
    scorer = BehaviorScorer(clock=FakeClock())
    for _ in range(5):  # never leaves → a single entry, never reaches 3
        _enter(scorer)
    r = scorer.score(1, _neutral(), PERSON_H, in_zones={"shelf"})
    assert "repeated_shelf_visit" not in r.behaviors


def test_no_zones_means_no_zone_criteria() -> None:
    scorer = BehaviorScorer(clock=FakeClock())
    _conceal(scorer)
    # No in_zones passed (camera has no zones) → zone criteria never fire.
    r = scorer.score(1, _neutral(), PERSON_H)
    assert "exit_after_concealment" not in r.behaviors
    assert "repeated_shelf_visit" not in r.behaviors
