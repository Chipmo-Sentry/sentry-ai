"""Behavior Engine v2 — state-machine + sequence-aware retail-theft scoring.

Supersedes the v1 6-dimension single-frame scorer. Key changes (ADR-0024):

  * **Absolute 0-100 score** (not the ADR-0022 normalize curve). Weights are
    scaled so a full concealment sequence ≈ CRITICAL. risk_pct = clamp(raw, 0,100).
  * **4 risk levels:** LOW 0-10 / MEDIUM 11-25 / HIGH 26-50 / CRITICAL 51-100.
  * **Per-track state machine:** IDLE → SUSPICIOUS → PRODUCT_INTERACTION →
    CONCEALMENT → ALERT (EXIT_ATTEMPT reserved — needs exit zones).
  * **Sequence engine:** ordered behavior patterns award bonus score and can
    drive the state forward — sequence scoring has HIGHER priority than any
    single detector (a completed concealment sequence jumps straight to ALERT).
  * **Keypoint-confidence aware** (#1): when keypoints carry a 3rd confidence
    column, low-confidence joints are ignored — far fewer false detections from
    occluded/uncertain joints. Falls back to (x,y)-only for legacy callers.
  * **Temporal smoothing** (#2): noisy single-frame dims (looking_around,
    body_block, crouch, rapid_movement) must persist N consecutive frames before
    they score — kills the "turned their head once" false positive.
  * **New detectors:** bag_interaction (wrist inside a bag bbox while holding)
    and pocket_interaction (wrist at a hip while holding).

Detectors live HERE; the backend `behaviors` catalog may also list criteria that
have no detector yet (repeated_shelf_visit, group_distraction, exit_after_
concealment, rfid_mismatch) — those are inert (score 0) until a detector ships.

Per-track state decays: 0.98 when idle, 0.999 when "holding".
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sentry_ai.live_worker.yolo_det import Item

# === COCO-17 keypoint indices ===
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12

# === Default raw weights (ADR-0024 absolute scale; DB-tunable via /api/v1/behaviors)
# Scaled to the v2 spec so the 0-100 score is meaningful directly:
#   picking up = normal shopping (modest); CONCEALMENT is determinative.
DEFAULT_WEIGHTS: dict[str, float] = {
    # LEVEL 1 — suspicious (don't alert alone)
    "looking_around": 2.0,
    "loitering": 3.0,
    "rapid_movement": 2.0,
    # LEVEL 2 — concealment indicators
    "item_pickup": 10.0,
    "body_block": 5.0,
    "crouch": 2.0,
    "wrist_to_torso": 12.0,
    "bag_interaction": 15.0,
    "pocket_interaction": 12.0,
}

SCORE_DECAY_IDLE = 0.98
SCORE_DECAY_HOLDING = 0.999
STALE_TRACK_SEC = 5.0

# Hold release (T06/H1): once `holding` is set, it must clear again when the
# person is no longer in contact with a held item. A track is released back to
# not-holding after this many consecutive analyzed frames with NO hold-supporting
# signal (wrist on an item bbox, or any concealment activity this frame).
# Without this, a single COCO pickup pins `holding=True` forever → permanent hold
# decay, the IDLE reset never fires, and bag/pocket/concealment detectors stay
# armed on a normal shopper. DB-tunable via the `engine` config.
HOLD_RELEASE_FRAMES = 30

# === Risk levels (absolute 0-100, ADR-0024) ===
LEVEL_LOW_MAX = 10.0  # 0-10   LOW    🟢
LEVEL_MEDIUM_MAX = 25.0  # 11-25  MEDIUM 🟡
LEVEL_HIGH_MAX = 50.0  # 26-50  HIGH   🔴 ; >50 CRITICAL 🔴

# Back-compat aliases (older imports / poller threshold path). Under v2 these are
# the green/yellow PERCENT cutoffs, not raw-score cutoffs.
DEFAULT_GREEN_MAX = LEVEL_LOW_MAX
DEFAULT_YELLOW_MAX = LEVEL_MEDIUM_MAX

# Temporal smoothing: a noisy single-frame dim must hold this many consecutive
# analyzed frames before it scores (and then scores once per window thereafter).
SMOOTH_FRAMES = 3
_SMOOTHED_DIMS = ("looking_around", "body_block", "crouch", "rapid_movement")

# Keypoint confidence gate (#1) — joints below this are treated as "not detected".
# Calibrated on YOLO11s-pose; YOLO26's RLE head shifts the confidence
# distribution (ADR-0026) — re-check against verified_cases before tightening.
MIN_KP_CONF = 0.3

# Loitering: same centroid (within radius) for this long → suspicious dwell.
LOITER_RADIUS_FRAC = 0.25  # of person height
LOITER_SECONDS = 30.0  # v2 spec default (was 8) — DB-tunable

# Sequence engine: an ordered pattern must complete within this span to fire.
SEQUENCE_WINDOW_SEC = 60.0
_EVENT_HISTORY_CAP = 64

# === Tunable engine params (ADR-0024 v2; hot-tuned from backend /api/v1/behaviors).
# Global knobs — the backend ships these in the `engine` object; the poller calls
# BehaviorScorer.update_params(). All have safe code defaults so a partial/missing
# config never breaks scoring.
DEFAULT_ENGINE: dict[str, float] = {
    "smooth_frames": float(
        SMOOTH_FRAMES
    ),  # consecutive frames a noisy dim must hold before it scores
    "decay_idle": SCORE_DECAY_IDLE,  # per-frame score decay when NOT holding an item
    "decay_holding": SCORE_DECAY_HOLDING,  # per-frame decay while holding an item (slower)
    "sequence_window_sec": SEQUENCE_WINDOW_SEC,  # window for an ordered pattern to complete
    "loiter_radius_frac": LOITER_RADIUS_FRAC,  # dwell radius as a fraction of person height
    "stale_track_sec": STALE_TRACK_SEC,  # drop a per-track state unseen this long
    "hold_release_frames": float(
        HOLD_RELEASE_FRAMES
    ),  # consecutive item-free frames before `holding` clears (T06/H1)
}

# Per-detector sensitivity params. `*_frac` are fractions of the person's height;
# `cadence` is a frame count; `hold_floor` a minimum score. Each criterion carries
# these in its catalog `params` object; missing keys fall back to these defaults.
DEFAULT_DETECTOR_PARAMS: dict[str, dict[str, float]] = {
    "looking_around": {"offset_frac": 0.15},
    "body_block": {"collapse_frac": 0.55, "ema_alpha": 0.1},
    "crouch": {"frac": 0.15, "hold_floor": 5.0},
    "wrist_to_torso": {"frac": 0.15, "cadence": 8.0},
    "pocket_interaction": {"radius_frac": 0.12},
    "rapid_movement": {"frac": 0.08},
    "loitering": {"seconds": LOITER_SECONDS},
}

RiskColor = Literal["green", "yellow", "red"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class BehaviorState(IntEnum):
    """Per-track behavior state (ordered — higher = more advanced toward theft)."""

    IDLE = 0
    SUSPICIOUS = 1
    PRODUCT_INTERACTION = 2
    CONCEALMENT = 3
    EXIT_ATTEMPT = 4  # reserved — needs exit zones (not yet emitted)
    ALERT = 5


# Behavior keys that mean "concealing a held item on the body".
_CONCEALMENT_KEYS = frozenset({"wrist_to_torso", "bag_interaction", "pocket_interaction"})
# Just the "hide it away" finishers (bag/pocket), excluding wrist_to_torso.
_HIDE_KEYS = frozenset({"bag_interaction", "pocket_interaction"})


@dataclass(frozen=True, slots=True)
class SequenceRule:
    """An ordered behavior pattern that awards a bonus when it completes.

    `pattern` items are behavior keys OR a meta-token:
      "concealment" → any of wrist_to_torso / bag_interaction / pocket_interaction
      "conceal_hide" → bag_interaction / pocket_interaction
    `critical=True` forces the track straight to ALERT when the rule completes
    (sequence priority over individual detectors).
    """

    key: str
    pattern: tuple[str, ...]
    bonus: float
    critical: bool = False


# v2 spec sequence rules (camera-feasible subset; group_distraction rule omitted —
# no multi-person detector yet).
DEFAULT_SEQUENCES: tuple[SequenceRule, ...] = (
    SequenceRule("seq_look_pickup", ("looking_around", "item_pickup"), 5.0),
    SequenceRule("seq_pickup_wrist", ("item_pickup", "wrist_to_torso"), 10.0),
    SequenceRule("seq_pickup_bag", ("item_pickup", "bag_interaction"), 15.0),
    SequenceRule(
        "seq_pickup_wrist_bag",
        ("item_pickup", "wrist_to_torso", "bag_interaction"),
        25.0,
    ),
    SequenceRule(
        "seq_loiter_pickup_conceal",
        ("loitering", "item_pickup", "concealment"),
        15.0,
    ),
    # LEVEL 4 critical event: pickup → wrist-to-torso → hide in bag/pocket.
    SequenceRule(
        "concealment_sequence",
        ("item_pickup", "wrist_to_torso", "conceal_hide"),
        30.0,
        critical=True,
    ),
)

_SEQ_TOKENS: dict[str, frozenset[str]] = {
    "concealment": _CONCEALMENT_KEYS,
    "conceal_hide": _HIDE_KEYS,
}


def risk_level(pct: float) -> RiskLevel:
    """Map a 0-100 risk_pct to the v2 4-level band (ADR-0024)."""
    if pct <= LEVEL_LOW_MAX:
        return "LOW"
    if pct <= LEVEL_MEDIUM_MAX:
        return "MEDIUM"
    if pct <= LEVEL_HIGH_MAX:
        return "HIGH"
    return "CRITICAL"


def classify_color(score: float, green_max: float, yellow_max: float) -> RiskColor:
    """3-color band for the existing wire/overlay (LOW→green, MEDIUM→yellow,
    HIGH+CRITICAL→red). Kept for back-compat; operates on the 0-100 pct."""
    if score <= green_max:
        return "green"
    if score <= yellow_max:
        return "yellow"
    return "red"


def clamp_pct(raw: float) -> float:
    """ADR-0024 absolute model: the raw accumulated score IS the 0-100 risk_pct."""
    if raw <= 0.0:
        return 0.0
    return min(100.0, raw)


def _kp_valid(kp: NDArray[np.float32], min_conf: float = MIN_KP_CONF) -> bool:
    """A keypoint is valid if x,y are real (>1.0) and — when a confidence column
    is present — its confidence clears `min_conf` (#1). Legacy (x,y)-only arrays
    pass on coordinates alone."""
    if not (kp[0] > 1.0 and kp[1] > 1.0):
        return False
    if kp.shape[0] >= 3:
        return bool(kp[2] >= min_conf)
    return True


@dataclass(slots=True)
class ScoreResult:
    """What `BehaviorScorer.score()` returns for one tracked person, one frame."""

    raw_score: float
    risk_pct: float
    color: RiskColor
    level: RiskLevel
    state: BehaviorState
    reasons: list[str]
    sequences: list[str]
    # Episode-accumulated criteria: stable keys in first-fired order with the
    # exact score contribution each has banked this episode (incl. sequence
    # bonuses). Resets with the episode (IDLE reset / stale cleanup).
    behaviors: list[str] = field(default_factory=list)
    behavior_scores: dict[str, float] = field(default_factory=dict)
    # Per-criterion seconds-from-episode-start (first firing), for the timeline.
    behavior_offsets: dict[str, float] = field(default_factory=dict)
    # Per-FIRE timeline: one entry per banking event ({key, offset_sec, score}),
    # chronological — drives the itemized alert breakdown (each +score increment
    # is its own row). Empty for manual uploads / older nodes.
    behavior_events: list[dict[str, Any]] = field(default_factory=list)
    episode_started_at: float | None = None


@dataclass(slots=True)
class TrackState:
    """Per-tracker memory between frames."""

    score: float = 0.0
    holding: bool = False
    # Consecutive analyzed frames with NO hold-supporting signal. Drives the
    # holding-latch release (T06/H1); reset to 0 whenever contact is re-seen.
    no_hold_frames: int = 0
    last_seen: float = field(default_factory=time.time)
    state: BehaviorState = BehaviorState.IDLE
    concealment_frames: int = 0
    avg_shoulder_w: float | None = None
    prev_l_wrist: tuple[float, float] | None = None
    prev_r_wrist: tuple[float, float] | None = None
    last_reasons: list[str] = field(default_factory=list)
    last_sequences: list[str] = field(default_factory=list)
    # Temporal smoothing: consecutive-frame counters per noisy dim.
    streak: dict[str, int] = field(default_factory=dict)
    # Sequence engine: ordered (behavior_key, ts) history + awarded rule keys.
    events: deque[tuple[str, float]] = field(
        default_factory=lambda: deque(maxlen=_EVENT_HISTORY_CAP)
    )
    awarded: set[str] = field(default_factory=set)
    # Loitering: anchor centroid + when the person settled there.
    loiter_anchor: tuple[float, float] | None = None
    loiter_since: float | None = None
    loiter_scored: bool = False
    # Episode tracking: when the first criterion of this episode fired, and the
    # accumulated score contribution per criterion key (insertion order =
    # first-fired order). Both reset on the IDLE reset in _advance_state.
    episode_started_at: float | None = None
    episode_behaviors: dict[str, float] = field(default_factory=dict)
    # First-fired wall-clock ts per criterion key this episode → drives the
    # per-criterion "X seconds in" offset on the alert timeline. Same reset.
    episode_behavior_ts: dict[str, float] = field(default_factory=dict)
    # Every banking event this episode: (key, wall-clock ts, banked amount), in
    # chronological order → the per-FIRE itemized breakdown. Same reset.
    episode_events: list[tuple[str, float, float]] = field(default_factory=list)


class BehaviorScorer:
    """Per-camera scorer with thread-safe per-track state.

    Instantiate one per CameraWorker. `score(tracker_id, keypoints, person_h,
    items)` advances the score and returns a `ScoreResult`.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # Injectable wall-clock source. Production uses time.time; tests pass a
        # controllable clock so episode/loiter/sequence timing is deterministic
        # and not at the mercy of how fast frames are processed under load.
        self._clock = clock
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        # v2: green/yellow are PERCENT cutoffs (LOW/MEDIUM, MEDIUM/HIGH).
        self.green_max: float = LEVEL_LOW_MAX
        self.yellow_max: float = LEVEL_MEDIUM_MAX
        self.high_max: float = LEVEL_HIGH_MAX
        # Tunable engine + per-detector params (hot-updated by the config poller).
        self.engine: dict[str, float] = dict(DEFAULT_ENGINE)
        self.det: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in DEFAULT_DETECTOR_PARAMS.items()
        }
        self.loiter_seconds: float = LOITER_SECONDS  # mirror of det["loitering"]["seconds"]
        self.sequences: tuple[SequenceRule, ...] = DEFAULT_SEQUENCES
        self._states: dict[int, TrackState] = {}
        self._lock = threading.Lock()

    # --- tunable-param accessors (cheap; called per frame) ---
    def _e(self, name: str) -> float:
        """Global engine param with a safe default."""
        return float(self.engine.get(name, DEFAULT_ENGINE[name]))

    def _dp(self, key: str, name: str, default: float) -> float:
        """Per-detector param with a safe default."""
        return float(self.det.get(key, {}).get(name, default))

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Hot-update weights from backend behavior config."""
        with self._lock:
            self.weights.update(new_weights)

    def update_params(
        self,
        engine: dict[str, float] | None = None,
        detector: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Hot-update engine + per-detector tuning params from backend config.

        Unknown engine keys are ignored; non-numeric values are skipped so a bad
        value can never crash the scorer. `smooth_frames`/`cadence` are coerced
        to a sane floor at read time, not here."""
        with self._lock:
            if engine:
                for k, v in engine.items():
                    if k in DEFAULT_ENGINE:
                        with contextlib.suppress(TypeError, ValueError):
                            self.engine[k] = float(v)
            if detector:
                for key, params in detector.items():
                    if not isinstance(params, dict):
                        continue
                    slot = self.det.setdefault(key, {})
                    for n, v in params.items():
                        with contextlib.suppress(TypeError, ValueError):
                            slot[n] = float(v)
            self.loiter_seconds = float(
                self.det.get("loitering", {}).get("seconds", LOITER_SECONDS)
            )

    def update_thresholds(
        self, green_max: float, yellow_max: float, red_max: float | None = None
    ) -> None:
        """Hot-update level cutoffs (percent) from backend behavior config.
        green_max = LOW/MEDIUM cutoff, yellow_max = MEDIUM/HIGH cutoff,
        red_max (optional) = HIGH/CRITICAL cutoff."""
        with self._lock:
            self.green_max = green_max
            self.yellow_max = yellow_max
            if red_max is not None:
                self.high_max = red_max

    def risk_pct(self, raw_score: float) -> float:
        """Convert a raw accumulated score to the 0-100 risk_pct (ADR-0024 absolute)."""
        return clamp_pct(raw_score)

    def level_for(self, pct: float) -> RiskLevel:
        """4-level band using THIS scorer's (tunable) thresholds — so a backend
        threshold change moves the level, not just the color."""
        if pct <= self.green_max:
            return "LOW"
        if pct <= self.yellow_max:
            return "MEDIUM"
        if pct <= self.high_max:
            return "HIGH"
        return "CRITICAL"

    def score(
        self,
        tracker_id: int,
        keypoints: NDArray[np.float32] | None,
        person_h: float,
        items: list[Item] | None = None,
    ) -> ScoreResult:
        """Compute and update score + state for one tracked person. Thread-safe.

        `keypoints` may be (17,2) [x,y] or (17,3) [x,y,conf] — confidence is used
        to gate joints when present. `items` is the nearby COCO item detections.
        """
        with self._lock:
            state = self._states.get(tracker_id)
            if state is None:
                state = TrackState()
                self._states[tracker_id] = state

            now = self._clock()
            state.last_seen = now

            # Decay before adding new evidence.
            decay = self._e("decay_holding") if state.holding else self._e("decay_idle")
            state.score = max(0.0, state.score * decay)

            delta = 0.0
            reasons: list[str] = []
            fired: set[str] = set()
            if keypoints is not None and len(keypoints) >= 17 and person_h > 0:
                delta, reasons, fired = self._analyze(state, keypoints, person_h, items or [], now)
            state.score += delta

            # First fired criterion since the last reset opens a new episode.
            if fired and state.episode_started_at is None:
                state.episode_started_at = now

            # Record fired behaviors into the sequence history (chronological).
            for key in fired:
                state.events.append((key, now))

            # Sequence engine — bonuses + critical jumps (higher priority).
            seq_bonus, seq_keys, seq_critical = self._match_sequences(state, now)
            state.score += seq_bonus
            for sk in seq_keys:
                reasons.append(f"Дараалал: {sk}")

            if reasons:
                state.last_reasons = reasons
            if seq_keys:
                state.last_sequences = list(seq_keys)

            pct = clamp_pct(state.score)
            self._advance_state(state, fired, pct, seq_critical)

            # Seconds from episode start for each fired criterion (0.0 before the
            # episode opens, which only happens with an empty ledger anyway).
            ep_start = state.episode_started_at
            offsets = (
                {
                    k: round(max(0.0, ts - ep_start), 1)
                    for k, ts in state.episode_behavior_ts.items()
                }
                if ep_start is not None
                else {}
            )
            # Per-FIRE breakdown: one {key, offset_sec, score} per banking event.
            events: list[dict[str, Any]] = (
                [
                    {
                        "key": k,
                        "offset_sec": round(max(0.0, ts - ep_start), 1),
                        "score": round(amount, 1),
                    }
                    for k, ts, amount in state.episode_events
                ]
                if ep_start is not None
                else []
            )

            return ScoreResult(
                raw_score=state.score,
                risk_pct=pct,
                color=classify_color(pct, self.green_max, self.yellow_max),
                level=self.level_for(pct),
                state=state.state,
                reasons=state.last_reasons,
                sequences=state.last_sequences,
                # Copies — ScoreResult escapes the lock; the ledger keeps mutating.
                behaviors=list(state.episode_behaviors.keys()),
                behavior_scores=dict(state.episode_behaviors),
                behavior_offsets=offsets,
                behavior_events=events,
                episode_started_at=state.episode_started_at,
            )

    def cleanup_stale(self) -> int:
        """Drop tracker states unseen for STALE_TRACK_SEC. Returns count removed."""
        now = self._clock()
        with self._lock:
            stale_sec = self._e("stale_track_sec")
            stale = [tid for tid, s in self._states.items() if now - s.last_seen > stale_sec]
            for tid in stale:
                del self._states[tid]
            return len(stale)

    # === Internal — single-frame heuristics ===

    def _smoothed(self, state: TrackState, key: str, active: bool) -> bool:
        """Temporal smoothing (#2): return True only when `active` has held for
        SMOOTH_FRAMES consecutive frames, then once per SMOOTH_FRAMES window."""
        if not active:
            state.streak[key] = 0
            return False
        n = state.streak.get(key, 0) + 1
        state.streak[key] = n
        sf = max(1, int(self._e("smooth_frames")))
        return n >= sf and (n % sf == 0)

    def _analyze(
        self,
        state: TrackState,
        kps: NDArray[np.float32],
        person_h: float,
        items: list[Item],
        now: float,
    ) -> tuple[float, list[str], set[str]]:
        l_eye = kps[KP_L_EYE]
        r_eye = kps[KP_R_EYE]
        l_shoulder = kps[KP_L_SHOULDER]
        r_shoulder = kps[KP_R_SHOULDER]
        l_wrist = kps[KP_L_WRIST]
        r_wrist = kps[KP_R_WRIST]
        l_hip = kps[KP_L_HIP]
        r_hip = kps[KP_R_HIP]

        shoulder_cx: float | None = None
        if _kp_valid(l_shoulder) and _kp_valid(r_shoulder):
            shoulder_cx = (l_shoulder[0] + r_shoulder[0]) / 2

        face_cx: float | None = None
        if _kp_valid(l_eye) and _kp_valid(r_eye):
            face_cx = (l_eye[0] + r_eye[0]) / 2

        shoulder_cy: float | None = None
        if _kp_valid(l_shoulder) and _kp_valid(r_shoulder):
            shoulder_cy = (l_shoulder[1] + r_shoulder[1]) / 2

        # Hip line. Desk/overhead cameras crop the hips off-frame, so the hip
        # keypoints are never valid → pocket/wrist_to_torso/crouch could NEVER
        # fire (concealment was undetectable on upper-body views). Fall back to a
        # waist line estimated below the shoulders from body proportions, so a
        # hand moving toward the lower body still registers.
        hip_cy: float | None = None
        if _kp_valid(l_hip) and _kp_valid(r_hip):
            hip_cy = (l_hip[1] + r_hip[1]) / 2
        elif shoulder_cy is not None:
            hip_cy = shoulder_cy + person_h * self._dp("concealment", "hip_estimate_frac", 0.5)

        delta = 0.0
        reasons: list[str] = []
        fired: set[str] = set()

        def _add(key: str, amount: float, reason: str) -> None:
            """Bank one criterion firing: score delta + episode ledger + reason."""
            nonlocal delta
            delta += amount
            state.episode_behaviors[key] = state.episode_behaviors.get(key, 0.0) + amount
            state.episode_behavior_ts.setdefault(key, now)  # first firing wins
            state.episode_events.append((key, now, amount))  # per-fire breakdown
            reasons.append(reason)
            fired.add(key)

        # 1. Looking around — face center drifts laterally from shoulder center.
        looking = (
            face_cx is not None
            and shoulder_cx is not None
            and abs(face_cx - shoulder_cx)
            > person_h * self._dp("looking_around", "offset_frac", 0.15)
        )
        if self._smoothed(state, "looking_around", looking):
            _add("looking_around", self.weights.get("looking_around", 0.0), "Орчноо харах")

        # 2. Item pickup — wrist enters a COCO item bbox. The first contact sets
        # holding=True; while holding, ongoing contact is the signal that keeps
        # the latch alive (its absence releases it — see step 10).
        wrist_on_item = False
        if items:
            for wrist in (l_wrist, r_wrist):
                if not _kp_valid(wrist):
                    continue
                matched: str | None = None
                # Tolerance: count the wrist NEAR the item, not strictly inside.
                # The hand grips an item's edge and item detection runs at ~2 fps
                # (slightly stale box), so strict containment made item_pickup
                # fire far too rarely → holding never set → concealment muted.
                m = person_h * self._dp("item_pickup", "margin_frac", 0.08)
                for it in items:
                    ix1, iy1, ix2, iy2 = it.box
                    if ix1 - m < wrist[0] < ix2 + m and iy1 - m < wrist[1] < iy2 + m:
                        matched = it.label
                        break
                if matched is not None:
                    wrist_on_item = True
                    if not state.holding:
                        state.holding = True
                        _add("item_pickup", self.weights.get("item_pickup", 0.0), f"{matched} авах")
                    break

        # 3. Body block — shoulder width collapses (turning back to camera).
        block = False
        if _kp_valid(l_shoulder) and _kp_valid(r_shoulder):
            shoulder_w = abs(r_shoulder[0] - l_shoulder[0])
            if state.avg_shoulder_w is None:
                state.avg_shoulder_w = float(shoulder_w)
            else:
                alpha = self._dp("body_block", "ema_alpha", 0.1)
                state.avg_shoulder_w = (
                    state.avg_shoulder_w * (1.0 - alpha) + float(shoulder_w) * alpha
                )
                block = shoulder_w < state.avg_shoulder_w * self._dp(
                    "body_block", "collapse_frac", 0.55
                )
        if self._smoothed(state, "body_block", block):
            _add("body_block", self.weights.get("body_block", 0.0), "Биеэр далдлах")

        # 4. Crouch — torso vertical extent collapses.
        crouch = False
        if hip_cy is not None and _kp_valid(l_shoulder):
            crouch = abs(hip_cy - l_shoulder[1]) < person_h * self._dp("crouch", "frac", 0.15)
        if self._smoothed(state, "crouch", crouch):
            if state.holding:
                _add(
                    "crouch",
                    max(self.weights.get("crouch", 0.0), self._dp("crouch", "hold_floor", 5.0)),
                    "Бөхийж бараа нуух",
                )
            else:
                _add("crouch", self.weights.get("crouch", 0.0), "Бөхийх")

        # 5. Hand to torso/chest — a wrist brought INTO the torso box: between the
        # shoulders horizontally, and from the shoulder line down to the waist
        # vertically. Catches chest / inner-pocket / under-arm concealment that's
        # visible on upper-body (desk/overhead) cameras where the hips are off
        # frame. NOT gated on `holding`: item detection is unreliable, and a hand
        # deliberately brought between the shoulders is itself the signal — arms
        # resting at the sides sit at/outside the shoulder x-edges, so they don't
        # match. Sustained via concealment_frames + cadence to avoid flicker.
        if shoulder_cy is not None and _kp_valid(l_shoulder) and _kp_valid(r_shoulder):
            sx_lo, sx_hi = sorted((float(l_shoulder[0]), float(r_shoulder[0])))
            torso_bot = hip_cy if hip_cy is not None else shoulder_cy + person_h * 0.5
            in_torso = any(
                _kp_valid(w) and sx_lo < w[0] < sx_hi and shoulder_cy < w[1] < torso_bot
                for w in (l_wrist, r_wrist)
            )
            if in_torso:
                state.concealment_frames += 1
                cadence = max(1, int(self._dp("wrist_to_torso", "cadence", 8.0)))
                if state.concealment_frames % cadence == 0:
                    _add(
                        "wrist_to_torso",
                        self.weights.get("wrist_to_torso", 0.0),
                        f"Гараа биедээ нуух ({state.concealment_frames}f)",
                    )
            else:
                state.concealment_frames = max(0, state.concealment_frames - 1)

        # 6. Bag interaction — wrist inside a bag bbox while holding merchandise.
        if state.holding and items:
            bags = [it for it in items if it.label in ("handbag", "backpack", "suitcase")]
            if bags and self._wrist_in_any(l_wrist, r_wrist, [b.box for b in bags]):
                _add("bag_interaction", self.weights.get("bag_interaction", 0.0), "Гар уут руу")

        # 7. Pocket interaction — wrist near a hip while holding. Real hip
        # keypoints win; when they're off-frame (upper-body cameras) fall back to
        # an estimated hip point below each shoulder at the waist line (hip_cy).
        if state.holding and hip_cy is not None:
            radius = person_h * self._dp("pocket_interaction", "radius_frac", 0.12)
            hip_points: list[tuple[float, float]] = []
            if _kp_valid(l_hip):
                hip_points.append((float(l_hip[0]), float(l_hip[1])))
            if _kp_valid(r_hip):
                hip_points.append((float(r_hip[0]), float(r_hip[1])))
            if not hip_points and _kp_valid(l_shoulder) and _kp_valid(r_shoulder):
                hip_points = [
                    (float(l_shoulder[0]), hip_cy),
                    (float(r_shoulder[0]), hip_cy),
                ]
            for hx, hy in hip_points:
                hit = any(
                    _kp_valid(w) and math.dist((float(w[0]), float(w[1])), (hx, hy)) < radius
                    for w in (l_wrist, r_wrist)
                )
                if hit:
                    _add(
                        "pocket_interaction",
                        self.weights.get("pocket_interaction", 0.0),
                        "Халаас руу",
                    )
                    break

        # 8. Rapid movement — wrist velocity spike; only when holding.
        rapid = False
        for wrist, prev_attr in ((l_wrist, "prev_l_wrist"), (r_wrist, "prev_r_wrist")):
            if not _kp_valid(wrist):
                continue
            prev: tuple[float, float] | None = getattr(state, prev_attr)
            if (
                prev is not None
                and state.holding
                and math.dist((float(wrist[0]), float(wrist[1])), prev)
                > person_h * self._dp("rapid_movement", "frac", 0.08)
            ):
                rapid = True
            setattr(state, prev_attr, (float(wrist[0]), float(wrist[1])))
        if self._smoothed(state, "rapid_movement", rapid):
            _add("rapid_movement", self.weights.get("rapid_movement", 0.0), "Хурдан хөдөлгөөн")

        # 9. Loitering — staying in roughly one spot for a long time.
        if shoulder_cx is not None and hip_cy is not None:
            centroid = (shoulder_cx, float(hip_cy))
            radius = person_h * self._e("loiter_radius_frac")
            if state.loiter_anchor is None or math.dist(centroid, state.loiter_anchor) > radius:
                state.loiter_anchor = centroid
                state.loiter_since = now
                state.loiter_scored = False
            elif (
                state.loiter_since is not None
                and not state.loiter_scored
                and (now - state.loiter_since) >= self.loiter_seconds
            ):
                _add("loitering", self.weights.get("loitering", 0.0), "Удаан зогсох")
                state.loiter_scored = True

        # 10. Hold-latch release (T06/H1) — clear `holding` once the person is no
        # longer in contact with a held item. A frame "supports" the hold if the
        # wrist is on an item bbox OR any concealment behavior fired this frame
        # (the item may be hidden, so concealment counts as still-holding).
        if state.holding:
            hold_signal = wrist_on_item or bool(fired & _CONCEALMENT_KEYS)
            if hold_signal:
                state.no_hold_frames = 0
            else:
                state.no_hold_frames += 1
                release_after = max(1, int(self._e("hold_release_frames")))
                if state.no_hold_frames >= release_after:
                    state.holding = False
                    state.no_hold_frames = 0
                    state.concealment_frames = 0

        return delta, reasons, fired

    @staticmethod
    def _wrist_in_any(
        l_wrist: NDArray[np.float32],
        r_wrist: NDArray[np.float32],
        boxes: list[tuple[float, float, float, float]],
    ) -> bool:
        for wrist in (l_wrist, r_wrist):
            if not _kp_valid(wrist):
                continue
            for x1, y1, x2, y2 in boxes:
                if x1 < wrist[0] < x2 and y1 < wrist[1] < y2:
                    return True
        return False

    # === Sequence engine ===

    def _match_sequences(self, state: TrackState, now: float) -> tuple[float, list[str], bool]:
        """Award bonuses for ordered patterns that completed within the window.
        Returns (bonus_total, awarded_keys_this_call, any_critical)."""
        # Prune events outside the window.
        cutoff = now - self._e("sequence_window_sec")
        while state.events and state.events[0][1] < cutoff:
            state.events.popleft()
        if not state.events:
            return 0.0, [], False

        bonus = 0.0
        awarded: list[str] = []
        critical = False
        for rule in self.sequences:
            if rule.key in state.awarded:
                continue
            if self._pattern_present(state.events, rule.pattern):
                # Bonus is catalog-tunable: a backend weight for the rule key
                # overrides the default; weight 0 (disabled criterion) skips it.
                bonus_val = self.weights.get(rule.key, rule.bonus)
                if bonus_val <= 0:
                    continue
                state.awarded.add(rule.key)
                bonus += bonus_val
                awarded.append(rule.key)
                # Sequence bonus joins the episode ledger like any criterion.
                state.episode_behaviors[rule.key] = (
                    state.episode_behaviors.get(rule.key, 0.0) + bonus_val
                )
                state.episode_behavior_ts.setdefault(rule.key, now)
                state.episode_events.append((rule.key, now, bonus_val))
                if rule.critical:
                    critical = True
        return bonus, awarded, critical

    @staticmethod
    def _pattern_present(events: deque[tuple[str, float]], pattern: tuple[str, ...]) -> bool:
        """True if `pattern` appears as an ordered subsequence of `events`.
        Pattern items may be concrete behavior keys or meta-tokens."""
        idx = 0
        for key, _ts in events:
            want = pattern[idx]
            token = _SEQ_TOKENS.get(want)
            if (token is not None and key in token) or key == want:
                idx += 1
                if idx == len(pattern):
                    return True
        return False

    # === State machine ===

    def _advance_state(
        self, state: TrackState, fired: set[str], pct: float, seq_critical: bool
    ) -> None:
        """Move the per-track state forward by this frame's evidence. The state
        never regresses except a full reset back to IDLE when the person calms
        down (score decays to LOW and they're no longer holding)."""
        target = state.state

        if fired & {"looking_around", "loitering", "rapid_movement"} or pct > self.green_max:
            target = max(target, BehaviorState.SUSPICIOUS)
        if state.holding or "item_pickup" in fired:
            target = max(target, BehaviorState.PRODUCT_INTERACTION)
        if fired & _CONCEALMENT_KEYS or state.concealment_frames > 0:
            target = max(target, BehaviorState.CONCEALMENT)
        # Sequence priority + CRITICAL band → ALERT, overriding individual score.
        if seq_critical or pct > self.high_max:
            target = BehaviorState.ALERT

        # Reset when the person has clearly calmed down — lets a fresh episode
        # re-trigger sequence bonuses later. A behavior whose temporal-smoothing
        # streak is still building counts as ongoing even on the frames between
        # its once-per-window emissions: don't close the episode out from under
        # an active behavior (that would fragment one continuous episode into a
        # fresh one every SMOOTH_FRAMES window).
        streak_active = any(n > 0 for n in state.streak.values())
        if pct <= self.green_max and not state.holding and not fired and not streak_active:
            target = BehaviorState.IDLE
            state.events.clear()
            state.awarded.clear()
            state.concealment_frames = 0
            state.episode_started_at = None
            state.episode_behaviors.clear()
            state.episode_behavior_ts.clear()
            state.episode_events.clear()

        state.state = target
