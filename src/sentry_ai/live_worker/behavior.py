"""6-dimensional behavior scoring — port of v1 Chipmo Security Camera AI.

Ref: D:/ai camera/Version 1/Chipmo_Security_Camera_AI/shoplift_detector/
     app/services/ai_service.py::_analyze_behavior

The scorer holds per-tracker state (history, decay) and produces a risk_pct
(0-100) + color band (green/yellow/red) for each tracked person every frame.

Dimensions (default weights):
  1. looking_around (1.5)   — face center vs shoulder center offset
  2. item_pickup (15.0)     — wrist enters shelf/item ROI (M1: skipped — no shelf
                              zones configured; will reach in L4.5)
  3. body_block (3.0)       — torso shoulder-width drops below rolling avg
  4. crouch (1.0)           — vertical torso height collapses
  5. wrist_to_torso (5.0)   — wrist held near hip (only when "holding")
  6. rapid_movement (1.5)   — wrist velocity spike (only when "holding")

Per-track state decays: 0.98 when idle, 0.999 when "holding" (M1: rarely set).
Score is clamped to 100 for risk_pct display.

For M1 we run WITHOUT shelf zones or COCO item detection. That means
`item_pickup` doesn't fire and `holding` stays False, which in turn means
dims 5 and 6 mostly stay quiet. Dims 1, 3, 4 still produce signal —
enough to demo "person acting suspicious" (lots of looking around +
crouching) trends.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
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

# === Defaults (M1; per-camera tuning in L4 follow-up) ===
DEFAULT_WEIGHTS: dict[str, float] = {
    "looking_around": 1.5,
    "item_pickup": 15.0,
    "body_block": 3.0,
    "crouch": 1.0,
    "wrist_to_torso": 5.0,
    "rapid_movement": 1.5,
}

SCORE_DECAY_IDLE = 0.98
SCORE_DECAY_HOLDING = 0.999
STALE_TRACK_SEC = 5.0

# Default ABSOLUTE accumulated-score thresholds (no longer percent).
# Overridden at runtime by backend's /api/v1/behaviors config (live tuning).
DEFAULT_GREEN_MAX = 5.0
DEFAULT_YELLOW_MAX = 15.0

RiskColor = Literal["green", "yellow", "red"]


def classify_color(score: float, green_max: float, yellow_max: float) -> RiskColor:
    if score < green_max:
        return "green"
    if score < yellow_max:
        return "yellow"
    return "red"


def _kp_valid(kp: NDArray[np.float32]) -> bool:
    """A keypoint is valid if x,y are above 1.0 (0,0 = not detected)."""
    return bool(kp[0] > 1.0 and kp[1] > 1.0)


@dataclass(slots=True)
class TrackState:
    """Per-tracker memory between frames."""

    score: float = 0.0
    holding: bool = False
    last_seen: float = field(default_factory=time.time)
    concealment_frames: int = 0
    avg_shoulder_w: float | None = None
    prev_l_wrist: tuple[float, float] | None = None
    prev_r_wrist: tuple[float, float] | None = None
    last_reasons: list[str] = field(default_factory=list)


class BehaviorScorer:
    """Per-camera scorer with thread-safe per-track state.

    Instantiate one per CameraWorker. `score(tracker_id, keypoints, person_h)`
    advances the score and returns (risk_pct, color, reasons).
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.green_max: float = DEFAULT_GREEN_MAX
        self.yellow_max: float = DEFAULT_YELLOW_MAX
        self._states: dict[int, TrackState] = {}
        self._lock = threading.Lock()

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Hot-update weights from backend behavior config."""
        with self._lock:
            self.weights.update(new_weights)

    def update_thresholds(self, green_max: float, yellow_max: float) -> None:
        """Hot-update color band thresholds from backend behavior config."""
        with self._lock:
            self.green_max = green_max
            self.yellow_max = yellow_max

    def score(
        self,
        tracker_id: int,
        keypoints: NDArray[np.float32] | None,
        person_h: float,
        items: list[Item] | None = None,
    ) -> tuple[float, RiskColor, list[str]]:
        """Compute and update score for one tracked person. Thread-safe.

        `items` is the list of nearby item detections (COCO model output, L4.5).
        Pass None or [] to skip item_pickup logic — leaves `holding` unchanged.

        Returns (raw_score, color, list of triggered reasons this frame).
        score is unbounded (was previously clamped to 100); color comes from
        absolute thresholds (green_max / yellow_max).
        """
        with self._lock:
            state = self._states.get(tracker_id)
            if state is None:
                state = TrackState()
                self._states[tracker_id] = state

            state.last_seen = time.time()

            # Score decay before adding new evidence
            decay = SCORE_DECAY_HOLDING if state.holding else SCORE_DECAY_IDLE
            state.score = max(0.0, state.score * decay)

            delta = 0.0
            reasons: list[str] = []
            if keypoints is not None and len(keypoints) >= 17 and person_h > 0:
                delta, reasons = self._analyze(state, keypoints, person_h, items or [])
            state.score += delta
            if reasons:
                state.last_reasons = reasons

            return state.score, classify_color(state.score, self.green_max, self.yellow_max), state.last_reasons

    def cleanup_stale(self) -> int:
        """Drop tracker states unseen for STALE_TRACK_SEC. Returns count removed."""
        now = time.time()
        with self._lock:
            stale = [
                tid for tid, s in self._states.items() if now - s.last_seen > STALE_TRACK_SEC
            ]
            for tid in stale:
                del self._states[tid]
            return len(stale)

    # === Internal — single-frame heuristic ===

    def _analyze(
        self,
        state: TrackState,
        kps: NDArray[np.float32],
        person_h: float,
        items: list[Item],
    ) -> tuple[float, list[str]]:
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

        # 1. Looking around — face center drifts laterally from shoulder center
        if face_cx is not None and shoulder_cx is not None:
            offset = abs(face_cx - shoulder_cx)
            if offset > person_h * 0.15:
                delta += self.weights["looking_around"]
                reasons.append("Орчноо харах")

        # 2. Item pickup — wrist enters COCO-detected item bbox.
        # Once fired, `holding=True` persists until track ages out, which
        # in turn activates dims 5 (wrist_to_torso) and 6 (rapid_movement).
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
                    delta += self.weights["item_pickup"]
                    reasons.append(f"{matched} авах")
                    break

        # 3. Body block — shoulder width collapses (turning back to camera)
        if _kp_valid(l_shoulder) and _kp_valid(r_shoulder):
            shoulder_w = abs(r_shoulder[0] - l_shoulder[0])
            if state.avg_shoulder_w is None:
                state.avg_shoulder_w = float(shoulder_w)
            else:
                state.avg_shoulder_w = state.avg_shoulder_w * 0.9 + float(shoulder_w) * 0.1
                if shoulder_w < state.avg_shoulder_w * 0.55:
                    delta += self.weights["body_block"]
                    reasons.append("Биеэр далдлах")

        # 4. Crouch — torso vertical extent collapses
        if hip_cy is not None and _kp_valid(l_shoulder):
            torso_h = abs(hip_cy - l_shoulder[1])
            if torso_h < person_h * 0.15:
                if state.holding:
                    delta += 5.0  # bigger when concealing while crouching
                    reasons.append("Бөхийж бараа нуух")
                else:
                    delta += self.weights["crouch"]
                    reasons.append("Бөхийх")

        # 5. Wrist to torso — only fires when holding (M1: rare)
        if state.holding and hip_cy is not None:
            for wrist in (l_wrist, r_wrist):
                if not _kp_valid(wrist):
                    continue
                wrist_to_hip = abs(wrist[1] - hip_cy)
                if wrist_to_hip < person_h * 0.15:
                    state.concealment_frames += 1
                    if state.concealment_frames % 8 == 0:
                        delta += self.weights["wrist_to_torso"]
                        reasons.append(
                            f"Хувцас доор нуух ({state.concealment_frames}f)",
                        )
                else:
                    state.concealment_frames = max(0, state.concealment_frames - 1)

        # 6. Rapid movement — only fires when holding (M1: rare)
        if state.holding:
            for wrist, prev_attr in (
                (l_wrist, "prev_l_wrist"),
                (r_wrist, "prev_r_wrist"),
            ):
                if not _kp_valid(wrist):
                    continue
                prev: tuple[float, float] | None = getattr(state, prev_attr)
                if prev is not None:
                    speed = math.dist((float(wrist[0]), float(wrist[1])), prev)
                    if speed > person_h * 0.08:
                        delta += self.weights["rapid_movement"]
                        reasons.append("Хурдан хөдөлгөөн")
                setattr(state, prev_attr, (float(wrist[0]), float(wrist[1])))

        return delta, reasons
