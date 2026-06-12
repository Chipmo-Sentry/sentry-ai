"""Episode tracking on BehaviorScorer (live criteria display + dynamic clips):
episode_started_at, the per-criterion episode_behaviors ledger, sequence bonus
banking, the IDLE reset, and the TrackPayload wire fields. Pure logic —
synthetic COCO-17 keypoints, no YOLO/torch.
"""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.behavior import (
    DEFAULT_WEIGHTS,
    SMOOTH_FRAMES,
    BehaviorScorer,
    BehaviorState,
)
from sentry_ai.live_worker.schemas import TrackPayload
from sentry_ai.live_worker.yolo_det import Item

# COCO-17 indices
L_EYE, R_EYE = 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

PERSON_H = 200.0


def _kp3() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


def _neutral() -> np.ndarray:
    k = _kp3()
    k[L_SHOULDER] = (90, 100, 1.0)
    k[R_SHOULDER] = (110, 100, 1.0)
    k[L_EYE] = (96, 60, 1.0)
    k[R_EYE] = (104, 60, 1.0)
    k[L_HIP] = (92, 300, 1.0)
    k[R_HIP] = (108, 300, 1.0)
    return k


def _looking() -> np.ndarray:
    k = _neutral()
    k[L_EYE] = (300, 60, 1.0)
    k[R_EYE] = (308, 60, 1.0)
    return k


def _pick_up(scorer: BehaviorScorer, tid: int = 1) -> None:
    k = _neutral()
    k[L_WRIST] = (200, 150, 1.0)
    item = Item(label="cell phone", box=(180, 130, 220, 170), score=0.9)
    scorer.score(tid, k, PERSON_H, items=[item])


def _fire_looking(scorer: BehaviorScorer, tid: int = 1):
    """Drive looking frames through temporal smoothing until it fires once."""
    r = None
    for _ in range(SMOOTH_FRAMES):
        r = scorer.score(tid, _looking(), PERSON_H)
    assert r is not None
    return r


# === episode start ===


def test_no_episode_before_any_criterion_fires() -> None:
    scorer = BehaviorScorer()
    r = scorer.score(1, _neutral(), PERSON_H)
    assert r.episode_started_at is None
    assert r.behaviors == []
    assert r.behavior_scores == {}


def test_episode_starts_on_first_fired_criterion() -> None:
    scorer = BehaviorScorer()
    r = _fire_looking(scorer)
    assert r.episode_started_at is not None
    assert r.behaviors == ["looking_around"]
    assert r.behavior_scores["looking_around"] == DEFAULT_WEIGHTS["looking_around"]


def test_episode_start_is_stable_across_later_firings() -> None:
    scorer = BehaviorScorer()
    first = _fire_looking(scorer)
    later = _fire_looking(scorer)  # fires again after another smoothing window
    assert later.episode_started_at == first.episode_started_at


# === ledger accumulation ===


def test_ledger_orders_keys_by_first_fired_and_accumulates() -> None:
    scorer = BehaviorScorer()
    _fire_looking(scorer)
    _pick_up(scorer)
    r = scorer.score(1, _neutral(), PERSON_H)
    # looking fired before pickup; the seq_look_pickup sequence completed too.
    assert r.behaviors[:2] == ["looking_around", "item_pickup"]
    assert r.behavior_scores["item_pickup"] == DEFAULT_WEIGHTS["item_pickup"]
    # Re-fire looking: same key accumulates, order unchanged.
    looking_before = r.behavior_scores["looking_around"]
    r2 = _fire_looking(scorer)
    assert r2.behaviors[:2] == ["looking_around", "item_pickup"]
    assert r2.behavior_scores["looking_around"] > looking_before


def test_sequence_bonus_lands_in_ledger() -> None:
    scorer = BehaviorScorer()
    _fire_looking(scorer)
    _pick_up(scorer)
    r = scorer.score(1, _neutral(), PERSON_H)
    assert "seq_look_pickup" in r.behaviors
    assert r.behavior_scores["seq_look_pickup"] > 0


def test_crouch_while_holding_banks_hold_floor() -> None:
    """The ledger must record the ACTUAL banked amount (hold_floor for crouch
    while holding), not the nominal catalog weight."""
    scorer = BehaviorScorer()
    _pick_up(scorer)
    crouched = _neutral()
    crouched[L_HIP] = (92, 120, 1.0)  # hips nearly at shoulder height
    crouched[R_HIP] = (108, 120, 1.0)
    r = None
    for _ in range(SMOOTH_FRAMES):
        r = scorer.score(1, crouched, PERSON_H)
    assert r is not None
    assert "crouch" in r.behaviors
    # hold_floor default 5.0 > crouch weight 2.0
    assert r.behavior_scores["crouch"] == 5.0


# === IDLE reset ===


def test_idle_reset_clears_episode() -> None:
    scorer = BehaviorScorer()
    r = _fire_looking(scorer)
    assert r.behaviors  # episode open
    # One calm frame: looking weight (2.0) keeps pct <= green_max, not holding,
    # nothing fired → the _advance_state IDLE reset closes the episode.
    r2 = scorer.score(1, _neutral(), PERSON_H)
    assert r2.state == BehaviorState.IDLE
    assert r2.episode_started_at is None
    assert r2.behaviors == []
    assert r2.behavior_scores == {}


def test_new_episode_after_reset_starts_fresh_ledger() -> None:
    scorer = BehaviorScorer()
    _fire_looking(scorer)
    scorer.score(1, _neutral(), PERSON_H)  # reset
    again = _fire_looking(scorer)
    # Episode reopened with a ledger that only holds the new firing (the old
    # 2.0 was wiped by the reset — NOT accumulated to 4.0).
    assert again.episode_started_at is not None
    assert again.behaviors == ["looking_around"]
    assert again.behavior_scores["looking_around"] == DEFAULT_WEIGHTS["looking_around"]


# === wire schema ===


def test_track_payload_carries_episode_fields() -> None:
    p = TrackPayload(
        person_id=1,
        box=(0, 0, 10, 10),
        det_confidence=0.9,
        behaviors=["looking_around", "item_pickup"],
        behavior_scores={"looking_around": 4.0, "item_pickup": 10.0},
        reasons=["Орчноо харах", "cell phone авах"],
        episode_started_ms=1760000000000,
    )
    dumped = p.model_dump(mode="json")
    assert dumped["behaviors"] == ["looking_around", "item_pickup"]
    assert dumped["behavior_scores"] == {"looking_around": 4.0, "item_pickup": 10.0}
    assert dumped["reasons"] == ["Орчноо харах", "cell phone авах"]
    assert dumped["episode_started_ms"] == 1760000000000


def test_track_payload_episode_fields_default_empty() -> None:
    """Older callers that don't pass the new fields still serialize cleanly."""
    p = TrackPayload(person_id=1, box=(0, 0, 10, 10), det_confidence=0.9)
    dumped = p.model_dump(mode="json")
    assert dumped["behaviors"] == []
    assert dumped["behavior_scores"] == {}
    assert dumped["reasons"] == []
    assert dumped["episode_started_ms"] is None
