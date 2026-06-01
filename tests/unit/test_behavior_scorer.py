"""BehaviorScorer — color bands, decay, weight/threshold hot-update, item pickup.

Pure logic: builds synthetic COCO-17 keypoint arrays, no YOLO/torch needed.
"""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.behavior import (
    DEFAULT_GREEN_MAX,
    DEFAULT_YELLOW_MAX,
    BehaviorScorer,
    classify_color,
)
from sentry_ai.live_worker.yolo_det import Item

# COCO-17 indices
L_EYE, R_EYE = 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


def _blank_kps() -> np.ndarray:
    """17x2 array, all (0,0) = 'not detected'."""
    return np.zeros((17, 2), dtype=np.float32)


def _neutral_person() -> np.ndarray:
    """A calmly-standing person — face centered over shoulders, normal torso."""
    kps = _blank_kps()
    # Shoulders centered at x=100
    kps[L_SHOULDER] = (90, 100)
    kps[R_SHOULDER] = (110, 100)
    # Eyes centered at x=100 too (no looking around)
    kps[L_EYE] = (96, 60)
    kps[R_EYE] = (104, 60)
    # Hips well below shoulders (tall torso, no crouch)
    kps[L_HIP] = (92, 300)
    kps[R_HIP] = (108, 300)
    return kps


# === classify_color ===


def test_classify_color_bands() -> None:
    assert classify_color(0.0, 5.0, 15.0) == "green"
    assert classify_color(4.9, 5.0, 15.0) == "green"
    assert classify_color(5.0, 5.0, 15.0) == "yellow"
    assert classify_color(14.9, 5.0, 15.0) == "yellow"
    assert classify_color(15.0, 5.0, 15.0) == "red"
    assert classify_color(100.0, 5.0, 15.0) == "red"


def test_default_thresholds_sane() -> None:
    assert DEFAULT_GREEN_MAX < DEFAULT_YELLOW_MAX


# === Score is unbounded (no clamp to 100) ===


def test_score_not_clamped_to_100() -> None:
    scorer = BehaviorScorer(weights={"looking_around": 50.0})
    kps = _neutral_person()
    # Force looking-around: move eyes far right of shoulders
    kps[L_EYE] = (300, 60)
    kps[R_EYE] = (308, 60)
    person_h = 200.0
    # Accumulate several frames
    last = 0.0
    for _ in range(5):
        last, _color, _reasons = scorer.score(1, kps, person_h)
    # 5 frames × 50 weight (minus decay) should well exceed 100
    assert last > 100.0


# === Decay ===


def test_idle_score_decays() -> None:
    scorer = BehaviorScorer()
    # Seed a score via looking-around, then feed blank frames (no triggers)
    kps = _neutral_person()
    kps[L_EYE] = (300, 60)
    kps[R_EYE] = (308, 60)
    scorer.score(1, kps, 200.0)
    seeded, _, _ = scorer.score(1, kps, 200.0)

    blank = _blank_kps()
    decayed, _, _ = scorer.score(1, blank, 200.0)
    assert decayed < seeded  # idle decay 0.98


# === Weight / threshold hot-update ===


def test_update_thresholds_changes_color() -> None:
    scorer = BehaviorScorer()
    # Manually set a known score via repeated looking_around
    kps = _neutral_person()
    kps[L_EYE] = (300, 60)
    kps[R_EYE] = (308, 60)
    score = 0.0
    for _ in range(3):
        score, _, _ = scorer.score(1, kps, 200.0)

    # With very high thresholds, same score is green
    scorer.update_thresholds(green_max=score + 100, yellow_max=score + 200)
    _, color_hi, _ = scorer.score(1, _blank_kps(), 200.0)
    assert color_hi == "green"

    # With very low thresholds, it's red
    scorer.update_thresholds(green_max=0.1, yellow_max=0.2)
    _, color_lo, _ = scorer.score(2, kps, 200.0)
    assert color_lo in ("yellow", "red")


def test_update_weights_affects_delta() -> None:
    low = BehaviorScorer(weights={"looking_around": 1.0})
    high = BehaviorScorer(weights={"looking_around": 10.0})
    kps = _neutral_person()
    kps[L_EYE] = (300, 60)
    kps[R_EYE] = (308, 60)
    s_low, _, _ = low.score(1, kps, 200.0)
    s_high, _, _ = high.score(1, kps, 200.0)
    assert s_high > s_low


# === item_pickup → holding ===


def test_item_pickup_sets_holding_and_scores() -> None:
    scorer = BehaviorScorer(weights={"item_pickup": 15.0})
    kps = _neutral_person()
    # Put left wrist at a known location
    kps[L_WRIST] = (200, 150)
    # An item bbox surrounding that wrist
    item = Item(label="cell phone", box=(180, 130, 220, 170), score=0.9)

    score, _color, reasons = scorer.score(1, kps, 200.0, items=[item])
    assert score >= 15.0 * 0.98  # weight added (minus one decay tick)
    assert any("авах" in r for r in reasons)


def test_no_item_no_pickup() -> None:
    scorer = BehaviorScorer(weights={"item_pickup": 15.0})
    kps = _neutral_person()
    kps[L_WRIST] = (200, 150)
    # Item far from wrist
    item = Item(label="bottle", box=(0, 0, 10, 10), score=0.9)
    score, _color, reasons = scorer.score(1, kps, 200.0, items=[item])
    assert not any("авах" in r for r in reasons)
    assert score == 0.0


# === Stale cleanup ===


def test_cleanup_stale_removes_old_tracks() -> None:
    scorer = BehaviorScorer()
    scorer.score(1, _neutral_person(), 200.0)
    scorer.score(2, _neutral_person(), 200.0)
    # Force last_seen far in the past
    import time

    for st in scorer._states.values():
        st.last_seen = time.time() - 999
    removed = scorer.cleanup_stale()
    assert removed == 2
    assert len(scorer._states) == 0


# === Missing keypoints don't crash ===


def test_blank_keypoints_no_score() -> None:
    scorer = BehaviorScorer()
    score, color, reasons = scorer.score(1, _blank_kps(), 200.0)
    assert score == 0.0
    assert color == "green"
    assert reasons == []


def test_none_keypoints_safe() -> None:
    scorer = BehaviorScorer()
    score, color, _ = scorer.score(1, None, 200.0)
    assert score == 0.0
    assert color == "green"
