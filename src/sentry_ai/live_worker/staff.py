"""Staff identification via neck-badge color (owner plan 2026-07-17).

Distinguishes staff from visitors WITHOUT biometrics: staff wear a lanyard +
badge in one distinctive color, configured centrally as `staff_badge_color`
(hex "#rrggbb" or a named color) on the ai-nodes config. Per analyzed frame a
tiny chest crop (between the shoulders, from YOLO-pose keypoints) is tested for
that color in HSV space — microseconds of CPU, viable on edge boxes.

Vote-lock, mirroring `demographics.py`: color hits accumulate per track; at
`min_hits` the track becomes a CANDIDATE and the VLM is asked once ("is this
person wearing a {color} staff badge/lanyard? yes/no") on a person crop — that
kills lookalike false positives (scarves, bags, printed shirts). A VLM "no"
locks the track as NOT staff. When the VLM is unreachable (cloud down, pure
edge box), color evidence alone locks at the stricter `color_only_hits`, so
staff labeling keeps working offline.

The lock is propagated to the store-global re-ID person
(`StorePersonRegistry.mark_staff`), so once ANY camera confirms the badge the
person stays staff across cameras and with their back turned — per the plan:
"үнэмлэх НЭГ удаа таарахад өдөржин ажилтан".

Alerts are NOT suppressed for staff — internal theft stays monitored; the flag
only recolors the live box and excludes staff from visitor analytics.

Privacy: person crops for the VLM question exist only in memory on their way to
the local VLM endpoint; nothing is written to disk or logged.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.staff")

Box = tuple[float, float, float, float]

# COCO-17 keypoint indices (matches yolo_runner.py / reid.py).
_L_SHOULDER, _R_SHOULDER = 5, 6
_KP_CONF_MIN = 0.35

# Named colors → HSV center (OpenCV H∈[0,180)). Tolerances below.
_NAMED_HSV: dict[str, int] = {
    "red": 0,
    "orange": 15,
    "yellow": 28,
    "green": 65,
    "cyan": 95,
    "blue": 112,
    "purple": 135,
    "pink": 168,
}
_HUE_TOL = 12  # ± around the center (red wraps)
_SAT_MIN = 90  # badge lanyards are saturated; grays/whites never match
_VAL_MIN = 60

# Forget a track's vote state this long after it was last seen.
_STATE_TTL_SEC = 60.0
_STATS_LOG_SEC = 60.0
# VLM person-crop max edge (px) — badge visibility needs little resolution.
_VLM_CROP_MAX_DIM = 448
_VLM_TIMEOUT_SEC = 25.0

_PROMPT = (
    "Look at the person in the image. Are they wearing a staff ID badge or "
    "lanyard around their neck in a {color} color? Answer strictly as JSON: "
    '{{"badge": true}} or {{"badge": false}}.'
)


def parse_badge_color(spec: str) -> tuple[int, int, int] | None:
    """Badge color spec → HSV center (h, s_min, v_min), or None when unparsable.

    Accepts a named color ("orange") or hex "#rrggbb". The named table is the
    recommended path — the owner picks a clearly-distinct lanyard color.
    """
    s = spec.strip().lower()
    if s in _NAMED_HSV:
        return (_NAMED_HSV[s], _SAT_MIN, _VAL_MIN)
    if s.startswith("#") and len(s) == 7:
        try:
            bgr = np.uint8([[[int(s[5:7], 16), int(s[3:5], 16), int(s[1:3], 16)]]])
        except ValueError:
            return None
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        # A near-gray hex (low saturation) can't be matched robustly — reject so
        # the operator picks a saturated color instead of getting noise.
        if int(hsv[1]) < _SAT_MIN:
            return None
        return (int(hsv[0]), _SAT_MIN, _VAL_MIN)
    return None


def _chest_box(box: Box, keypoints: NDArray[np.float32] | None) -> Box:
    """Upper-chest region where a neck badge hangs.

    With confident shoulder keypoints: the band between the shoulders,
    extending from the shoulder line down ~55% of the shoulder-width (badges
    hang just below the collarbone). Fallback (no pose): central 60% width,
    18%–45% of the person-box height.
    """
    x1, y1, x2, y2 = box
    if keypoints is not None and keypoints.shape[0] > _R_SHOULDER:
        ls, rs = keypoints[_L_SHOULDER], keypoints[_R_SHOULDER]
        if float(ls[2]) >= _KP_CONF_MIN and float(rs[2]) >= _KP_CONF_MIN:
            sx1, sx2 = sorted((float(ls[0]), float(rs[0])))
            sy = (float(ls[1]) + float(rs[1])) / 2.0
            sw = max(8.0, sx2 - sx1)
            # Slight horizontal inset — lanyard straps converge toward the sternum.
            inset = sw * 0.12
            return (sx1 + inset, sy, sx2 - inset, sy + sw * 0.55)
    bw, bh = x2 - x1, y2 - y1
    return (x1 + bw * 0.20, y1 + bh * 0.18, x2 - bw * 0.20, y1 + bh * 0.45)


def _color_frac(frame_bgr: NDArray[np.uint8], chest: Box, hsv_target: tuple[int, int, int]) -> float:
    """Fraction of chest pixels within the badge-color HSV window (red wraps)."""
    h, w = frame_bgr.shape[:2]
    cx1 = max(0, int(chest[0]))
    cy1 = max(0, int(chest[1]))
    cx2 = min(w, int(chest[2]))
    cy2 = min(h, int(chest[3]))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return 0.0
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat_min, val_min = hsv_target
    lo_h, hi_h = hue - _HUE_TOL, hue + _HUE_TOL
    lo = np.array([max(0, lo_h), sat_min, val_min], dtype=np.uint8)
    hi = np.array([min(179, hi_h), 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    if lo_h < 0:  # red wrap: also match the top of the hue circle
        mask |= cv2.inRange(
            hsv,
            np.array([180 + lo_h, sat_min, val_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
    elif hi_h > 179:
        mask |= cv2.inRange(
            hsv,
            np.array([0, sat_min, val_min], dtype=np.uint8),
            np.array([hi_h - 180, 255, 255], dtype=np.uint8),
        )
    return float(np.count_nonzero(mask)) / float(mask.size)


class _TrackLike(Protocol):
    """The slice of TrackedDetection this module needs (keeps imports light)."""

    @property
    def tracker_id(self) -> int: ...
    @property
    def box(self) -> Box: ...
    @property
    def keypoints(self) -> NDArray[np.float32] | None: ...


@dataclass(slots=True)
class _TrackState:
    hits: int = 0
    attempts: int = 0
    last_attempt_frame: int = -(10**9)  # first sighting is immediately eligible
    last_seen: float = 0.0
    # None = undecided; True/False = locked (VLM verdict, or color-only lock).
    locked: bool | None = None
    vlm_asked: bool = False


class _VlmGate:
    """Single background thread asking the VLM one badge question per candidate.

    The camera worker thread must never block on the VLM (seconds), so
    candidates are enqueued as (key, jpeg) and verdicts come back via a
    thread-safe dict the worker polls on its next frame. Queue is bounded —
    when the VLM is slow, extra candidates simply wait for a later cadence
    tick (or lock color-only).
    """

    def __init__(self, color_name: str) -> None:
        self.color_name = color_name
        self._q: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=8)
        self._verdicts: dict[int, bool] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="staff-vlm", daemon=True)
        self._thread.start()

    def submit(self, key: int, jpeg: bytes) -> bool:
        try:
            self._q.put_nowait((key, jpeg))
        except queue.Full:
            return False
        return True

    def poll(self, key: int) -> bool | None:
        with self._lock:
            return self._verdicts.pop(key, None)

    def _run(self) -> None:
        from sentry_ai.providers.oneshot import ask_json  # noqa: PLC0415 — import cycle

        while True:
            key, jpeg = self._q.get()
            reply = ask_json(
                _PROMPT.format(color=self.color_name), jpeg, timeout_sec=_VLM_TIMEOUT_SEC
            )
            if reply is None or not isinstance(reply.get("badge"), bool):
                continue  # unreachable/unparseable → no verdict; color-only path decides
            with self._lock:
                self._verdicts[key] = bool(reply["badge"])


class TrackStaff:
    """Per-track badge-color vote cache + VLM confirmation gate.

    observe() is called once per analyzed frame from the camera worker thread
    (single-threaded per instance) and returns tracker_id → is_staff for ALL
    tracks. False until a track's vote locks True.
    """

    def __init__(
        self,
        every_n: int = 8,
        max_per_frame: int = 3,
        frac_threshold: float = 0.08,
        min_hits: int = 3,
        color_only_hits: int = 8,
        max_attempts: int = 60,
        vlm_verify: bool = True,
    ) -> None:
        self.every_n = max(1, every_n)
        self.max_per_frame = max(1, max_per_frame)
        self.frac_threshold = frac_threshold
        self.min_hits = max(1, min_hits)
        self.color_only_hits = max(self.min_hits, color_only_hits)
        self.max_attempts = max_attempts
        self.vlm_verify = vlm_verify
        self._states: dict[int, _TrackState] = {}
        self._gate: _VlmGate | None = None
        # The parsed color currently in effect; re-parsed when config changes.
        self._color_spec: str | None = None
        self._hsv: tuple[int, int, int] | None = None
        self._attempts = 0
        self._hits_total = 0
        self._last_stats_log = time.monotonic()

    def _sync_color(self) -> bool:
        """Re-read the central badge color; True when staff detection is active."""
        from sentry_ai.runtime_config import get_staff_badge_color  # noqa: PLC0415

        spec = get_staff_badge_color()
        if spec != self._color_spec:
            self._color_spec = spec
            self._hsv = parse_badge_color(spec) if spec else None
            if spec and self._hsv is None:
                log.warning("staff.bad_badge_color", spec=spec)
            # Color changed → old votes are about the wrong color.
            self._states.clear()
            if self._hsv is not None and self.vlm_verify:
                self._gate = _VlmGate(spec or "")
        return self._hsv is not None

    def observe(
        self,
        frame_bgr: NDArray[np.uint8],
        tracked: Sequence[_TrackLike],
        frame_idx: int,
    ) -> dict[int, bool]:
        if not self._sync_color():
            return dict.fromkeys((t.tracker_id for t in tracked), False)
        assert self._hsv is not None
        now = time.monotonic()
        due: list[tuple[_TrackState, _TrackLike]] = []
        for t in tracked:
            st = self._states.get(t.tracker_id)
            if st is None:
                st = _TrackState()
                self._states[t.tracker_id] = st
            st.last_seen = now
            # Collect any pending VLM verdict first — it settles the track.
            if st.locked is None and st.vlm_asked and self._gate is not None:
                verdict = self._gate.poll(t.tracker_id)
                if verdict is not None:
                    st.locked = verdict
            if (
                st.locked is None
                and st.attempts < self.max_attempts
                and frame_idx - st.last_attempt_frame >= self.every_n
            ):
                due.append((st, t))
        due.sort(key=lambda d: d[0].last_attempt_frame)

        for st, t in due[: self.max_per_frame]:
            st.last_attempt_frame = frame_idx
            st.attempts += 1
            self._attempts += 1
            frac = _color_frac(frame_bgr, _chest_box(t.box, t.keypoints), self._hsv)
            if frac < self.frac_threshold:
                continue
            st.hits += 1
            self._hits_total += 1
            if st.hits < self.min_hits:
                continue
            # Candidate. VLM confirms when available; color-only locks at the
            # stricter threshold so an offline edge box still labels staff.
            if self.vlm_verify and self._gate is not None and not st.vlm_asked:
                jpeg = _person_jpeg(frame_bgr, t.box)
                if jpeg is not None and self._gate.submit(t.tracker_id, jpeg):
                    st.vlm_asked = True
                    continue
            if st.hits >= self.color_only_hits:
                st.locked = True
                log.info("staff.color_only_lock", hits=st.hits)

        self._log_stats(now)
        return {t.tracker_id: self._states[t.tracker_id].locked is True for t in tracked}

    def prune(self, now: float | None = None) -> None:
        ts = time.monotonic() if now is None else now
        for tid in [k for k, s in self._states.items() if ts - s.last_seen > _STATE_TTL_SEC]:
            del self._states[tid]

    def _log_stats(self, now: float) -> None:
        if now - self._last_stats_log < _STATS_LOG_SEC or self._attempts == 0:
            return
        self._last_stats_log = now
        staff = sum(1 for s in self._states.values() if s.locked is True)
        log.info(
            "staff.stats",
            attempts=self._attempts,
            color_hits=self._hits_total,
            staff_tracks=staff,
            tracked_states=len(self._states),
        )
        self._attempts = 0
        self._hits_total = 0


def _person_jpeg(frame_bgr: NDArray[np.uint8], box: Box) -> bytes | None:
    """Downscaled person crop as JPEG for the one-shot VLM question."""
    h, w = frame_bgr.shape[:2]
    pad_x = (box[2] - box[0]) * 0.08
    x1 = max(0, int(box[0] - pad_x))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2] + pad_x))
    y2 = min(h, int(box[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = min(1.0, _VLM_CROP_MAX_DIM / max(ch, cw))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return bytes(buf) if ok else None


def make_track_staff() -> TrackStaff | None:
    """Build the per-camera staff vote cache from settings, or None when disabled.

    Even when enabled this is inert (zero per-frame cost beyond a dict lookup)
    until a central `staff_badge_color` is configured.
    """
    from sentry_ai.settings import get_settings  # noqa: PLC0415 — avoid import cycle

    s = get_settings()
    if not s.staff_enabled:
        return None
    return TrackStaff(
        every_n=s.staff_every_n,
        max_per_frame=s.staff_max_per_frame,
        frac_threshold=s.staff_frac_threshold,
        min_hits=s.staff_min_hits,
        color_only_hits=s.staff_color_only_hits,
        max_attempts=s.staff_max_attempts,
        vlm_verify=s.staff_vlm_verify,
    )
