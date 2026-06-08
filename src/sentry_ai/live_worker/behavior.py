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

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

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

# === Risk levels (absolute 0-100, ADR-0024) ===
LEVEL_LOW_MAX = 10.0  # 0-10   LOW    🟢
LEVEL_MEDIUM_MAX = 25.0  # 11-25  MEDIUM 🟡
LEVEL_HIGH_MAX = 50.0  # 26-50  HIGH   🔴 ; >50 CRITICAL 🔴

# Back-compat aliases (older imports / poller threshold path). Under v2 these are
# the green/yellow PERCENT cutoffs, not raw-score cutoffs.
DEFAULT_GREEN_MAX = LEVEL_LOW_MAX
DEFAULT_YELLOW_MAX = LEVEL_MEDIUM_MAX
DEFAULT_RED_MAX = 100.0

# Temporal smoothing: a noisy single-frame dim must hold this many consecutive
# analyzed frames before it scores (and then scores once per window thereafter).
SMOOTH_FRAMES = 3
_SMOOTHED_DIMS = ("looking_around", "body_block", "crouch", "rapid_movement")

# Keypoint confidence gate (#1) — joints below this are treated as "not detected".
MIN_KP_CONF = 0.3

# Loitering: same centroid (within radius) for this long → suspicious dwell.
LOITER_RADIUS_FRAC = 0.25  # of person height
LOITER_SECONDS = 30.0  # v2 spec default (was 8) — DB-tunable

# Sequence engine: an ordered pattern must complete within this span to fire.
SEQUENCE_WINDOW_SEC = 60.0
_EVENT_HISTORY_CAP = 64

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


def normalize_pct(
    raw: float,
    green_max: float = 5.0,
    yellow_max: float = 15.0,
    red_max: float = 30.0,
) -> float:
    """Legacy ADR-0022 piecewise-linear normalize (retained for back-compat).

    The v2 engine uses absolute clamping (`clamp_pct`) instead, but this is kept
    so older callers / tests that import it keep working.
    """
    if raw <= 0 or green_max <= 0:
        return 0.0
    if raw <= green_max:
        return 30.0 * raw / green_max
    if raw <= yellow_max:
        span = max(yellow_max - green_max, 1e-6)
        return 30.0 + 40.0 * (raw - green_max) / span
    if raw <= red_max:
        span = max(red_max - yellow_max, 1e-6)
        return 70.0 + 30.0 * (raw - yellow_max) / span
    return 100.0


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


@dataclass(slots=True)
class TrackState:
    """Per-tracker memory between frames."""

    score: float = 0.0
    holding: bool = False
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


class BehaviorScorer:
    """Per-camera scorer with thread-safe per-track state.

    Instantiate one per CameraWorker. `score(tracker_id, keypoints, person_h,
    items)` advances the score and returns a `ScoreResult`.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        # v2: green/yellow are PERCENT cutoffs (LOW/MEDIUM, MEDIUM/HIGH).
        self.green_max: float = LEVEL_LOW_MAX
        self.yellow_max: float = LEVEL_MEDIUM_MAX
        self.high_max: float = LEVEL_HIGH_MAX
        self.loiter_seconds: float = LOITER_SECONDS
        self.sequences: tuple[SequenceRule, ...] = DEFAULT_SEQUENCES
        self._states: dict[int, TrackState] = {}
        self._lock = threading.Lock()

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Hot-update weights from backend behavior config."""
        with self._lock:
            self.weights.update(new_weights)

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

            now = time.time()
            state.last_seen = now

            # Decay before adding new evidence.
            decay = SCORE_DECAY_HOLDING if state.holding else SCORE_DECAY_IDLE
            state.score = max(0.0, state.score * decay)

            delta = 0.0
            reasons: list[str] = []
            fired: set[str] = set()
            if keypoints is not None and len(keypoints) >= 17 and person_h > 0:
                delta, reasons, fired = self._analyze(state, keypoints, person_h, items or [], now)
            state.score += delta

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

            return ScoreResult(
                raw_score=state.score,
                risk_pct=pct,
                color=classify_color(pct, self.green_max, self.yellow_max),
                level=self.level_for(pct),
                state=state.state,
                reasons=state.last_reasons,
                sequences=state.last_sequences,
            )

    def cleanup_stale(self) -> int:
        """Drop tracker states unseen for STALE_TRACK_SEC. Returns count removed."""
        now = time.time()
        with self._lock:
            stale = [tid for tid, s in self._states.items() if now - s.last_seen > STALE_TRACK_SEC]
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
        return n >= SMOOTH_FRAMES and (n % SMOOTH_FRAMES == 0)

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

        hip_cy: float | None = None
        if _kp_valid(l_hip) and _kp_valid(r_hip):
            hip_cy = (l_hip[1] + r_hip[1]) / 2

        delta = 0.0
        reasons: list[str] = []
        fired: set[str] = set()

        # 1. Looking around — face center drifts laterally from shoulder center.
        looking = (
            face_cx is not None
            and shoulder_cx is not None
            and abs(face_cx - shoulder_cx) > person_h * 0.15
        )
        if self._smoothed(state, "looking_around", looking):
            delta += self.weights.get("looking_around", 0.0)
            reasons.append("Орчноо харах")
            fired.add("looking_around")

        # 2. Item pickup — wrist enters a COCO item bbox → holding=True (persists).
        if items and not state.holding:
            for wrist in (l_wrist, r_wrist):
                if not _kp_valid(wrist):
                    continue
                matched: str | None = None
                for it in items:
                    ix1, iy1, ix2, iy2 = it.box
                    if ix1 < wrist[0] < ix2 and iy1 < wrist[1] < iy2:
                        matched = it.label
                        break
                if matched is not None:
                    state.holding = True
                    delta += self.weights.get("item_pickup", 0.0)
                    reasons.append(f"{matched} авах")
                    fired.add("item_pickup")
                    break

        # 3. Body block — shoulder width collapses (turning back to camera).
        block = False
        if _kp_valid(l_shoulder) and _kp_valid(r_shoulder):
            shoulder_w = abs(r_shoulder[0] - l_shoulder[0])
            if state.avg_shoulder_w is None:
                state.avg_shoulder_w = float(shoulder_w)
            else:
                state.avg_shoulder_w = state.avg_shoulder_w * 0.9 + float(shoulder_w) * 0.1
                block = shoulder_w < state.avg_shoulder_w * 0.55
        if self._smoothed(state, "body_block", block):
            delta += self.weights.get("body_block", 0.0)
            reasons.append("Биеэр далдлах")
            fired.add("body_block")

        # 4. Crouch — torso vertical extent collapses.
        crouch = False
        if hip_cy is not None and _kp_valid(l_shoulder):
            crouch = abs(hip_cy - l_shoulder[1]) < person_h * 0.15
        if self._smoothed(state, "crouch", crouch):
            if state.holding:
                delta += max(self.weights.get("crouch", 0.0), 5.0)
                reasons.append("Бөхийж бараа нуух")
            else:
                delta += self.weights.get("crouch", 0.0)
                reasons.append("Бөхийх")
            fired.add("crouch")

        # 5. Wrist to torso — wrist held near hip line; only when holding.
        if state.holding and hip_cy is not None:
            near_torso = False
            for wrist in (l_wrist, r_wrist):
                if not _kp_valid(wrist):
                    continue
                if abs(wrist[1] - hip_cy) < person_h * 0.15:
                    near_torso = True
                    break
            if near_torso:
                state.concealment_frames += 1
                if state.concealment_frames % 8 == 0:
                    delta += self.weights.get("wrist_to_torso", 0.0)
                    reasons.append(f"Хувцас доор нуух ({state.concealment_frames}f)")
                    fired.add("wrist_to_torso")
            else:
                state.concealment_frames = max(0, state.concealment_frames - 1)

        # 6. Bag interaction — wrist inside a bag bbox while holding merchandise.
        if state.holding and items:
            bags = [it for it in items if it.label in ("handbag", "backpack", "suitcase")]
            if bags and self._wrist_in_any(l_wrist, r_wrist, [b.box for b in bags]):
                delta += self.weights.get("bag_interaction", 0.0)
                reasons.append("Гар уут руу")
                fired.add("bag_interaction")

        # 7. Pocket interaction — wrist at a hip keypoint while holding.
        if state.holding:
            radius = person_h * 0.12
            for hip in (l_hip, r_hip):
                if not _kp_valid(hip):
                    continue
                hit = False
                for wrist in (l_wrist, r_wrist):
                    if (
                        _kp_valid(wrist)
                        and math.dist(
                            (float(wrist[0]), float(wrist[1])), (float(hip[0]), float(hip[1]))
                        )
                        < radius
                    ):
                        hit = True
                        break
                if hit:
                    delta += self.weights.get("pocket_interaction", 0.0)
                    reasons.append("Халаас руу")
                    fired.add("pocket_interaction")
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
                and math.dist((float(wrist[0]), float(wrist[1])), prev) > person_h * 0.08
            ):
                rapid = True
            setattr(state, prev_attr, (float(wrist[0]), float(wrist[1])))
        if self._smoothed(state, "rapid_movement", rapid):
            delta += self.weights.get("rapid_movement", 0.0)
            reasons.append("Хурдан хөдөлгөөн")
            fired.add("rapid_movement")

        # 9. Loitering — staying in roughly one spot for a long time.
        if shoulder_cx is not None and hip_cy is not None:
            centroid = (shoulder_cx, float(hip_cy))
            radius = person_h * LOITER_RADIUS_FRAC
            if state.loiter_anchor is None or math.dist(centroid, state.loiter_anchor) > radius:
                state.loiter_anchor = centroid
                state.loiter_since = now
                state.loiter_scored = False
            elif (
                state.loiter_since is not None
                and not state.loiter_scored
                and (now - state.loiter_since) >= self.loiter_seconds
            ):
                delta += self.weights.get("loitering", 0.0)
                reasons.append("Удаан зогсох")
                fired.add("loitering")
                state.loiter_scored = True

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
        cutoff = now - SEQUENCE_WINDOW_SEC
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
        # re-trigger sequence bonuses later.
        if pct <= self.green_max and not state.holding and not fired:
            target = BehaviorState.IDLE
            state.events.clear()
            state.awarded.clear()
            state.concealment_frames = 0

        state.state = target
