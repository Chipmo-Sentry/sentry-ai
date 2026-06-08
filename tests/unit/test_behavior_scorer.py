"""Behavior Engine v2 — levels, state machine, sequences, confidence gating,
temporal smoothing, bag/pocket detectors. Pure logic: synthetic COCO-17
keypoints, no YOLO/torch needed.
"""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.behavior import (
    DEFAULT_GREEN_MAX,
    DEFAULT_YELLOW_MAX,
    MIN_KP_CONF,
    SMOOTH_FRAMES,
    BehaviorScorer,
    BehaviorState,
    ScoreResult,
    clamp_pct,
    classify_color,
    risk_level,
)
from sentry_ai.live_worker.yolo_det import Item

# COCO-17 indices
L_EYE, R_EYE = 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

PERSON_H = 200.0


def _kp3() -> np.ndarray:
    """17x3 array [x, y, conf], all zero = 'not detected'."""
    return np.zeros((17, 3), dtype=np.float32)


def _neutral() -> np.ndarray:
    """A calmly-standing person — face centered over shoulders, tall torso,
    all keypoints high-confidence."""
    k = _kp3()
    k[L_SHOULDER] = (90, 100, 1.0)
    k[R_SHOULDER] = (110, 100, 1.0)
    k[L_EYE] = (96, 60, 1.0)
    k[R_EYE] = (104, 60, 1.0)
    k[L_HIP] = (92, 300, 1.0)
    k[R_HIP] = (108, 300, 1.0)
    return k


def _looking() -> np.ndarray:
    """Neutral but face swung far right of the shoulders."""
    k = _neutral()
    k[L_EYE] = (300, 60, 1.0)
    k[R_EYE] = (308, 60, 1.0)
    return k


# === level / color / clamp helpers ===


def test_risk_level_bands() -> None:
    assert risk_level(0.0) == "LOW"
    assert risk_level(10.0) == "LOW"
    assert risk_level(10.1) == "MEDIUM"
    assert risk_level(25.0) == "MEDIUM"
    assert risk_level(25.1) == "HIGH"
    assert risk_level(50.0) == "HIGH"
    assert risk_level(50.1) == "CRITICAL"
    assert risk_level(100.0) == "CRITICAL"


def test_classify_color_on_pct() -> None:
    assert classify_color(0.0, 10.0, 25.0) == "green"
    assert classify_color(10.0, 10.0, 25.0) == "green"
    assert classify_color(10.1, 10.0, 25.0) == "yellow"
    assert classify_color(25.0, 10.0, 25.0) == "yellow"
    assert classify_color(25.1, 10.0, 25.0) == "red"


def test_clamp_pct_caps_0_100() -> None:
    assert clamp_pct(-5.0) == 0.0
    assert clamp_pct(30.0) == 30.0
    assert clamp_pct(150.0) == 100.0


def test_default_thresholds_sane() -> None:
    assert DEFAULT_GREEN_MAX < DEFAULT_YELLOW_MAX


# === absolute scoring + decay ===


def test_score_returns_result_and_is_absolute() -> None:
    scorer = BehaviorScorer(weights={"looking_around": 50.0})
    k = _looking()
    last: ScoreResult | None = None
    for _ in range(SMOOTH_FRAMES):
        last = scorer.score(1, k, PERSON_H)
    assert last is not None
    # 50 weight fired once smoothing satisfied → well into the bands, capped 100.
    assert 0.0 < last.risk_pct <= 100.0
    assert last.risk_pct == clamp_pct(last.raw_score)


def test_idle_score_decays() -> None:
    scorer = BehaviorScorer(weights={"looking_around": 20.0})
    k = _looking()
    for _ in range(SMOOTH_FRAMES):
        seeded = scorer.score(1, k, PERSON_H)
    decayed = scorer.score(1, _kp3(), PERSON_H)  # blank → no triggers, decays
    assert decayed.raw_score < seeded.raw_score


# === temporal smoothing (#2) ===


def test_looking_around_needs_consecutive_frames() -> None:
    scorer = BehaviorScorer()
    k = _looking()
    # First SMOOTH_FRAMES-1 frames: not enough persistence → no score.
    for _ in range(SMOOTH_FRAMES - 1):
        r = scorer.score(1, k, PERSON_H)
        assert "Орчноо харах" not in r.reasons
    # On the SMOOTH_FRAMES-th consecutive frame it fires.
    r = scorer.score(1, k, PERSON_H)
    assert "Орчноо харах" in r.reasons


def test_intermittent_looking_never_fires() -> None:
    scorer = BehaviorScorer()
    look, calm = _looking(), _neutral()
    fired = False
    for i in range(8):
        r = scorer.score(1, look if i % 2 == 0 else calm, PERSON_H)
        fired = fired or ("Орчноо харах" in r.reasons)
    assert not fired  # streak keeps resetting


# === keypoint confidence gating (#1) ===


def test_low_confidence_joints_ignored() -> None:
    scorer = BehaviorScorer()
    k = _looking()
    k[L_EYE] = (300, 60, 0.1)  # below MIN_KP_CONF
    k[R_EYE] = (308, 60, 0.1)
    assert MIN_KP_CONF > 0.1
    for _ in range(SMOOTH_FRAMES + 2):
        r = scorer.score(1, k, PERSON_H)
    assert "Орчноо харах" not in r.reasons  # eyes gated out → no face center


def test_legacy_xy_keypoints_still_work() -> None:
    """(17,2) arrays (no confidence column) must still score on coordinates."""
    scorer = BehaviorScorer()
    k = _looking()[:, :2].copy()  # drop confidence column
    for _ in range(SMOOTH_FRAMES):
        r = scorer.score(1, k, PERSON_H)
    assert "Орчноо харах" in r.reasons


# === item pickup → holding → PRODUCT_INTERACTION ===


def test_item_pickup_sets_state_and_scores() -> None:
    scorer = BehaviorScorer()
    k = _neutral()
    k[L_WRIST] = (200, 150, 1.0)
    item = Item(label="cell phone", box=(180, 130, 220, 170), score=0.9)
    r = scorer.score(1, k, PERSON_H, items=[item])
    assert any("авах" in x for x in r.reasons)
    assert r.state >= BehaviorState.PRODUCT_INTERACTION


def test_no_item_no_pickup() -> None:
    scorer = BehaviorScorer()
    k = _neutral()
    k[L_WRIST] = (200, 150, 1.0)
    item = Item(label="bottle", box=(0, 0, 10, 10), score=0.9)
    r = scorer.score(1, k, PERSON_H, items=[item])
    assert not any("авах" in x for x in r.reasons)
    assert r.raw_score == 0.0
    assert r.state == BehaviorState.IDLE


# === bag / pocket detectors ===


def _pick_up(scorer: BehaviorScorer, tid: int = 1) -> None:
    """Drive one pickup frame so the track is `holding`."""
    k = _neutral()
    k[L_WRIST] = (200, 150, 1.0)
    item = Item(label="cell phone", box=(180, 130, 220, 170), score=0.9)
    scorer.score(tid, k, PERSON_H, items=[item])


def test_bag_interaction_when_holding() -> None:
    scorer = BehaviorScorer()
    _pick_up(scorer)
    k = _neutral()
    k[L_WRIST] = (400, 250, 1.0)
    bag = Item(label="handbag", box=(380, 230, 440, 290), score=0.8)
    r = scorer.score(1, k, PERSON_H, items=[bag])
    assert "Гар уут руу" in r.reasons


def test_pocket_interaction_when_holding() -> None:
    scorer = BehaviorScorer()
    _pick_up(scorer)
    k = _neutral()
    k[L_WRIST] = (92, 300, 1.0)  # right on the left hip keypoint
    r = scorer.score(1, k, PERSON_H, items=[])
    assert "Халаас руу" in r.reasons


# === sequence engine ===


def test_pickup_then_wrist_awards_sequence_bonus() -> None:
    scorer = BehaviorScorer()
    _pick_up(scorer)
    # 8 near-torso frames → one wrist_to_torso event (fires on the 8th).
    k = _neutral()
    k[L_WRIST] = (200, 300, 1.0)  # at hip Y, far in X (no pocket)
    last = None
    for _ in range(8):
        last = scorer.score(1, k, PERSON_H, items=[])
    assert last is not None
    assert "seq_pickup_wrist" in last.sequences


def test_concealment_sequence_is_critical_alert() -> None:
    scorer = BehaviorScorer()
    _pick_up(scorer)
    k = _neutral()
    k[L_WRIST] = (200, 300, 1.0)
    for _ in range(8):  # → wrist_to_torso event
        scorer.score(1, k, PERSON_H, items=[])
    # Now hide in the pocket (conceal_hide finisher).
    kp = _neutral()
    kp[L_WRIST] = (92, 300, 1.0)
    r = scorer.score(1, kp, PERSON_H, items=[])
    assert "concealment_sequence" in r.sequences
    assert r.state == BehaviorState.ALERT
    assert r.level == "CRITICAL"


# === state machine progression ===


def test_state_progresses_idle_to_suspicious() -> None:
    scorer = BehaviorScorer()
    k = _looking()
    last = None
    for _ in range(SMOOTH_FRAMES):
        last = scorer.score(1, k, PERSON_H)
    assert last is not None
    assert last.state >= BehaviorState.SUSPICIOUS


def test_state_resets_to_idle_when_calm() -> None:
    scorer = BehaviorScorer(weights={"looking_around": 4.0})
    k = _looking()
    for _ in range(SMOOTH_FRAMES):
        scorer.score(1, k, PERSON_H)
    # Feed many calm blank frames → decays to LOW, not holding → IDLE.
    last = None
    for _ in range(60):
        last = scorer.score(1, _kp3(), PERSON_H)
    assert last is not None
    assert last.state == BehaviorState.IDLE


# === weight / threshold hot-update ===


def test_update_weights_affects_score() -> None:
    low = BehaviorScorer(weights={"looking_around": 1.0})
    high = BehaviorScorer(weights={"looking_around": 10.0})
    k = _looking()
    rl = rh = None
    for _ in range(SMOOTH_FRAMES):
        rl = low.score(1, k, PERSON_H)
        rh = high.score(1, k, PERSON_H)
    assert rh.raw_score > rl.raw_score


def test_update_thresholds_changes_level_cutoffs() -> None:
    scorer = BehaviorScorer()
    scorer.update_thresholds(green_max=40.0, yellow_max=60.0, red_max=80.0)
    assert scorer.green_max == 40.0
    assert scorer.yellow_max == 60.0
    assert scorer.high_max == 80.0


def test_level_reflects_tuned_thresholds() -> None:
    """Regression (review M1/M2): the emitted level must follow the scorer's
    tunable thresholds, not fixed module constants."""
    scorer = BehaviorScorer()
    # Default: pct 30 → HIGH (25 < 30 ≤ 50).
    assert scorer.level_for(30.0) == "HIGH"
    # Raise the cutoffs → the same pct is now MEDIUM.
    scorer.update_thresholds(green_max=20.0, yellow_max=40.0, red_max=70.0)
    assert scorer.level_for(30.0) == "MEDIUM"
    assert scorer.level_for(80.0) == "CRITICAL"


# === robustness ===


def test_cleanup_stale_removes_old_tracks() -> None:
    import time

    scorer = BehaviorScorer()
    scorer.score(1, _neutral(), PERSON_H)
    scorer.score(2, _neutral(), PERSON_H)
    for st in scorer._states.values():
        st.last_seen = time.time() - 999
    assert scorer.cleanup_stale() == 2
    assert len(scorer._states) == 0


def test_blank_keypoints_no_score() -> None:
    scorer = BehaviorScorer()
    r = scorer.score(1, _kp3(), PERSON_H)
    assert r.raw_score == 0.0
    assert r.color == "green"
    assert r.level == "LOW"
    assert r.reasons == []


def test_none_keypoints_safe() -> None:
    scorer = BehaviorScorer()
    r = scorer.score(1, None, PERSON_H)
    assert r.raw_score == 0.0
    assert r.state == BehaviorState.IDLE
