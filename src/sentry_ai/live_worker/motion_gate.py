"""Motion gate — stop burning GPU on an empty, still scene.

A store camera spends most of the night (and much of a slow day) looking at a
scene where nothing moves and nobody is present. Running the full YOLO pass on
every frame_skip-th frame there is pure waste — on the cloud node it steals
headroom from the VLM, and on the edge Jetson boxes (8 cameras on a 67-TOPS
Nano) it is the difference between fitting and not fitting.

The gate closes ONLY when the scene is simultaneously:
  * still   — the downscaled gray absdiff shows no moving pixels, AND
  * empty   — the tracker currently has zero active tracks.
A person standing motionless keeps the gate open via their track; a person
ENTERING opens it via motion on the very frame they appear, so nothing is
missed. While closed, YOLO still runs every `idle_stride`-th frame (~1/s) as a
safety net and so health metrics never read as a dead worker.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

# Downscale width for the diff — big enough to see a person-sized change,
# small enough to cost microseconds.
_DIFF_W = 160
# A pixel counts as "moving" when its gray delta exceeds this (0-255). Absorbs
# sensor noise and slow illumination drift.
_PIXEL_DELTA = 14


@dataclass(slots=True)
class GateDecision:
    infer: bool  # run YOLO on this frame?
    idle: bool  # is the gate currently in idle (power-save) mode?


class MotionGate:
    def __init__(
        self,
        enabled: bool = True,
        area_frac: float = 0.002,
        quiet_after: int = 25,
        idle_stride: int = 8,
    ) -> None:
        self.enabled = enabled
        # Fraction of pixels that must move to count as scene motion. 0.002 of a
        # 160-wide frame ≈ a hand-sized blob — below a person entering, above
        # compression shimmer.
        self.area_frac = area_frac
        # Consecutive still+empty frames before the gate closes (warm-down) —
        # covers tracker coast-out so a briefly-occluded person can re-acquire.
        self.quiet_after = quiet_after
        # While idle, YOLO still runs every Nth frame as a safety net.
        self.idle_stride = max(2, idle_stride)
        self._prev: NDArray[np.uint8] | None = None
        self._still_streak = 0
        self._idle_counter = 0
        self.gated_total = 0  # frames where YOLO was skipped (for stats logs)

    def observe(self, frame_bgr: NDArray[np.uint8], active_tracks: int) -> GateDecision:
        """Decide whether this frame needs the detector. Call once per
        frame_skip-passed frame, BEFORE running YOLO."""
        if not self.enabled:
            return GateDecision(infer=True, idle=False)

        moving = self._motion_frac(frame_bgr) > self.area_frac
        if moving or active_tracks > 0:
            self._still_streak = 0
            self._idle_counter = 0
            return GateDecision(infer=True, idle=False)

        self._still_streak += 1
        if self._still_streak < self.quiet_after:
            return GateDecision(infer=True, idle=False)

        # Idle: still scene, no tracks, warm-down elapsed → thin the detector.
        self._idle_counter += 1
        if self._idle_counter % self.idle_stride == 0:
            return GateDecision(infer=True, idle=True)
        self.gated_total += 1
        return GateDecision(infer=False, idle=True)

    def _motion_frac(self, frame_bgr: NDArray[np.uint8]) -> float:
        h, w = frame_bgr.shape[:2]
        scale = _DIFF_W / max(1, w)
        small = cv2.resize(frame_bgr, (_DIFF_W, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev, self._prev = self._prev, gray
        if prev is None or prev.shape != gray.shape:
            return 1.0  # first frame / resolution change → treat as motion
        delta = cv2.absdiff(gray, prev)
        return float(np.count_nonzero(delta > _PIXEL_DELTA)) / delta.size
