"""Behavior Engine v2 — levels, state machine, sequences, confidence gating,
temporal smoothing, bag/pocket detectors. Pure logic: synthetic COCO-17
keypoints, no YOLO/torch needed.
"""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.behavior import (
    DEFAULT_GREEN_MAX,
    DEFAULT_WEIGHTS,
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
#
# Since the docs/33 P0-3 banking fix, bag/pocket bank through a TIME-based
# contact gate (min_hold_sec=0.4 of continuous contact, then once per
# interval_sec, capped per stint) — so these tests drive a controllable clock
# and SUSTAIN the contact pose instead of asserting on a single frame.


def _clocked_scorer(
    weights: dict[str, float] | None = None,
) -> tuple[BehaviorScorer, list[float]]:
    """Scorer with a synthetic, manually-advanced wall clock."""
    t = [0.0]
    return BehaviorScorer(weights, clock=lambda: t[0]), t


def _sustain(
    scorer: BehaviorScorer,
    t: list[float],
    kp: np.ndarray,
    items: list[Item],
    *,
    seconds: float = 0.8,
    dt: float = 0.2,
    tid: int = 1,
) -> ScoreResult:
    """Score the same pose repeatedly while advancing the clock; last result."""
    r: ScoreResult | None = None
    for _ in range(max(1, int(seconds / dt))):
        t[0] += dt
        r = scorer.score(tid, kp, PERSON_H, items=items)
    assert r is not None
    return r


def _pick_up(scorer: BehaviorScorer, tid: int = 1) -> None:
    """Drive one pickup frame so the track is `holding`."""
    k = _neutral()
    k[L_WRIST] = (200, 150, 1.0)
    item = Item(label="cell phone", box=(180, 130, 220, 170), score=0.9)
    scorer.score(tid, k, PERSON_H, items=[item])


def test_bag_interaction_when_holding() -> None:
    scorer, t = _clocked_scorer()
    _pick_up(scorer)
    k = _neutral()
    k[L_WRIST] = (400, 250, 1.0)
    bag = Item(label="handbag", box=(380, 230, 440, 290), score=0.8)
    r = _sustain(scorer, t, k, [bag])  # 0.8 s of contact > min_hold_sec
    assert "Гар уут руу" in r.reasons


def test_pocket_interaction_when_holding() -> None:
    scorer, t = _clocked_scorer()
    _pick_up(scorer)
    k = _neutral()
    k[L_WRIST] = (92, 300, 1.0)  # right on the left hip keypoint
    r = _sustain(scorer, t, k, [])
    assert "Халаас руу" in r.reasons


def test_pocket_interaction_fires_without_holding_by_default() -> None:
    """The fix: pocketing a NON-COCO retail item leaves `holding` False (no COCO
    pickup), yet hand-to-pocket is itself the concealment signal — so it must
    fire on geometry alone (require_holding defaults to 0)."""
    scorer, t = _clocked_scorer()
    k = _neutral()
    k[L_WRIST] = (92, 300, 1.0)  # on the left hip, NO prior pickup
    r = _sustain(scorer, t, k, [])
    assert "Халаас руу" in r.reasons
    assert scorer._states[1].holding is False  # fired with no COCO item pickup


def test_single_frame_wrist_at_hip_does_not_bank() -> None:
    """docs/33 P0-3: an incidental wrist-passes-hip (single frame, < min_hold_sec)
    must NOT bank — this was the per-frame-banking false-positive engine."""
    scorer, _t = _clocked_scorer()
    k = _neutral()
    k[L_WRIST] = (92, 300, 1.0)
    r = scorer.score(1, k, PERSON_H, items=[])
    assert "Халаас руу" not in r.reasons
    assert r.risk_pct == 0.0


def test_resting_hand_at_hip_never_saturates() -> None:
    """docs/33 P0-3 REGRESSION: a person standing with a hand resting at the hip
    for a full minute must plateau at ~weight × max_banks_per_contact (MEDIUM-ish),
    NOT ratchet to 100. Pre-fix this hit 100 in ~2 s (the PoseLift AUC-0.39 root
    cause: every benign clip peaked at 100, indistinguishable from theft)."""
    scorer, t = _clocked_scorer()
    k = _neutral()
    k[L_WRIST] = (92, 300, 1.0)  # resting on the hip, continuously
    peak = 0.0
    for _ in range(300):  # 60 s at 5 fps
        t[0] += 0.2
        r = scorer.score(1, k, PERSON_H, items=[])
        peak = max(peak, r.risk_pct)
    cap = DEFAULT_WEIGHTS["pocket_interaction"] * 3  # max_banks_per_contact
    assert peak <= cap + 10, f"peak {peak} — resting hand must not saturate"
    assert peak < 70, "resting hand must stay below the HIGH visual band"


def test_repeated_pocket_reaches_rebank_per_stint() -> None:
    """A REAL repeated pocketing motion (contact broken between reaches) banks
    again on each new stint — the gate caps dwell, not distinct reaches."""
    scorer, t = _clocked_scorer()
    reach = _neutral()
    reach[L_WRIST] = (92, 300, 1.0)
    away = _neutral()
    away[L_WRIST] = (60, 180, 1.0)  # hand away from the hip
    for _ in range(3):  # three distinct reaches, ~3 s apart
        _sustain(scorer, t, reach, [], seconds=0.8)
        _sustain(scorer, t, away, [], seconds=2.2)
    st = scorer._states[1]
    banked = st.episode_behaviors.get("pocket_interaction", 0.0)
    assert banked >= DEFAULT_WEIGHTS["pocket_interaction"] * 2, (
        f"distinct reaches must re-bank (got {banked})"
    )


def test_require_holding_param_restores_strict_pocket_gate() -> None:
    """Operators can set the per-detector `require_holding` param to 1 to demand a
    prior pickup again (the pre-fix strict behaviour)."""
    scorer = BehaviorScorer()
    scorer.update_params(detector={"pocket_interaction": {"require_holding": 1}})
    k = _neutral()
    k[L_WRIST] = (92, 300, 1.0)  # on the hip, but no pickup → strict gate blocks it
    r = scorer.score(1, k, PERSON_H, items=[])
    assert "Халаас руу" not in r.reasons


# === hold-latch release (T06/H1) ===


def test_holding_releases_after_item_free_frames() -> None:
    """Once a track picks up an item it is `holding`; after HOLD_RELEASE_FRAMES
    consecutive frames with no item contact and no concealment, `holding` clears."""
    scorer = BehaviorScorer()
    _pick_up(scorer)
    st = scorer._states[1]
    assert st.holding is True
    release_after = int(scorer._e("hold_release_frames"))
    # Feed neutral frames with no items and wrist away from hips/bags.
    k = _neutral()
    k[L_WRIST] = (150, 90, 1.0)  # mid-air, not near hip / not on any item
    k[R_WRIST] = (50, 90, 1.0)
    for _ in range(release_after):
        scorer.score(1, k, PERSON_H, items=[])
    assert st.holding is False
    assert st.concealment_frames == 0


def test_holding_persists_while_in_contact() -> None:
    """Ongoing wrist-on-item contact keeps the latch alive past the release window."""
    scorer = BehaviorScorer()
    release_after = int(scorer._e("hold_release_frames"))
    k = _neutral()
    k[L_WRIST] = (200, 150, 1.0)
    item = Item(label="cell phone", box=(180, 130, 220, 170), score=0.9)
    for _ in range(release_after + 5):
        scorer.score(1, k, PERSON_H, items=[item])
    assert scorer._states[1].holding is True


def test_released_track_can_return_to_idle() -> None:
    """After the hold latch releases, a calmed-down track resets to IDLE — the
    bug (T06/H1) was that `holding` stayed True forever and blocked this reset."""
    scorer = BehaviorScorer()
    _pick_up(scorer)
    k = _neutral()
    k[L_WRIST] = (150, 90, 1.0)
    k[R_WRIST] = (50, 90, 1.0)
    last = None
    # Enough frames to both release the latch and decay the score back to LOW.
    for _ in range(200):
        last = scorer.score(1, k, PERSON_H, items=[])
    assert last is not None
    assert scorer._states[1].holding is False
    assert last.state == BehaviorState.IDLE


# === sequence engine ===


def test_pickup_then_wrist_awards_sequence_bonus() -> None:
    scorer = BehaviorScorer()
    _pick_up(scorer)
    # 8 near-torso frames → one wrist_to_torso event (fires on the 8th).
    k = _neutral()
    k[L_WRIST] = (100, 200, 1.0)  # chest: between shoulders, below shoulder line (no pocket)
    last = None
    for _ in range(8):
        last = scorer.score(1, k, PERSON_H, items=[])
    assert last is not None
    assert "seq_pickup_wrist" in last.sequences


def test_concealment_sequence_is_critical_alert() -> None:
    scorer, t = _clocked_scorer()
    _pick_up(scorer)
    k = _neutral()
    k[L_WRIST] = (100, 200, 1.0)  # chest (between shoulders) → wrist_to_torso
    for _ in range(8):  # → wrist_to_torso event
        t[0] += 0.2
        scorer.score(1, k, PERSON_H, items=[])
    # Now hide in the pocket (conceal_hide finisher) — sustained past the
    # contact gate's min_hold_sec (docs/33 P0-3).
    kp = _neutral()
    kp[L_WRIST] = (92, 300, 1.0)
    r = _sustain(scorer, t, kp, [])
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


def test_update_params_changes_engine_and_detector() -> None:
    """Hot-tuning engine + per-detector params takes effect and is bad-value safe."""
    s = BehaviorScorer()
    assert s._e("smooth_frames") == 3.0
    assert s._dp("loitering", "seconds", 0.0) == 30.0
    s.update_params(
        engine={"smooth_frames": 8, "decay_idle": 0.95},
        detector={"loitering": {"seconds": 45}, "looking_around": {"offset_frac": 0.25}},
    )
    assert s._e("smooth_frames") == 8.0
    assert s._e("decay_idle") == 0.95
    assert s.loiter_seconds == 45.0
    assert s._dp("looking_around", "offset_frac", 0.0) == 0.25
    # Unknown engine key ignored; non-numeric value skipped (no crash).
    s.update_params(engine={"bogus": 1, "decay_idle": "nope"})  # type: ignore[dict-item]
    assert "bogus" not in s.engine
    assert s._e("decay_idle") == 0.95
