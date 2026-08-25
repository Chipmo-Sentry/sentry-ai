"""Mannequin filter — a "person" that never moves is not a visitor.

YOLO happily detects shop mannequins as persons, so they inflate visitor
counts, sit in the demographics, and can even accumulate behavior score while
"loitering" at a shelf forever. Two complementary rules mark a track as a
mannequin:

  1. ZONE — its foot stands inside an operator-drawn «mannequin» fixture on the
     floor plan (deterministic; the owner already maps display dummies), or
  2. STILLNESS — its foot hasn't moved more than ``move_frac`` of the frame for
     ``still_after_sec`` (mannequins outside drawn zones; nothing alive stands
     THAT still for minutes).

The latch RELEASES the moment the track genuinely moves (a re-dressed dummy
being carried, or a person misjudged after standing very still) — from then on
they count and score again, so the worst case of a false latch is a gap in an
unusually motionless shopper's counting, never a permanent loss.
"""

from __future__ import annotations

import math
import time


class MannequinFilter:
    def __init__(self, still_after_sec: float = 180.0, move_frac: float = 0.03) -> None:
        # A track must stay within `move_frac` (of frame size, normalized coords)
        # of its anchor for `still_after_sec` before the stillness rule latches.
        self.still_after_sec = still_after_sec
        self.move_frac = move_frac
        self._anchor: dict[int, tuple[float, float, float]] = {}  # tid → (x, y, since)
        self._latched: set[int] = set()
        self._last_seen: dict[int, float] = {}

    def observe(
        self,
        tid: int,
        foot_nx: float,
        foot_ny: float,
        in_zones: set[str] | None,
        now: float | None = None,
    ) -> bool:
        """Update with this frame's foot point; True = treat as mannequin."""
        now = time.monotonic() if now is None else now
        self._last_seen[tid] = now

        # Rule 1: standing in a drawn mannequin fixture → mannequin, immediately.
        if in_zones and "mannequin" in in_zones:
            self._latched.add(tid)
            self._anchor[tid] = (foot_nx, foot_ny, now)
            return True

        anchor = self._anchor.get(tid)
        if anchor is None:
            self._anchor[tid] = (foot_nx, foot_ny, now)
            return False
        ax, ay, since = anchor
        moved = math.hypot(foot_nx - ax, foot_ny - ay) > self.move_frac

        if tid in self._latched:
            if moved:
                # It walks → it's no mannequin (anymore). Release and re-anchor.
                self._latched.discard(tid)
                self._anchor[tid] = (foot_nx, foot_ny, now)
                return False
            return True

        if moved:
            self._anchor[tid] = (foot_nx, foot_ny, now)
            return False
        # Rule 2: perfectly still past the threshold → latch.
        if now - since >= self.still_after_sec:
            self._latched.add(tid)
            return True
        return False

    def prune(self, ttl_sec: float = 120.0, now: float | None = None) -> None:
        """Drop state for tracks unseen past `ttl_sec` (tracker ids recycle)."""
        now = time.monotonic() if now is None else now
        stale = [t for t, ts in self._last_seen.items() if now - ts > ttl_sec]
        for t in stale:
            self._last_seen.pop(t, None)
            self._anchor.pop(t, None)
            self._latched.discard(t)
