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

Stillness is banked per SPOT, not only per track: dark/noisy footage makes the
detector blink on a mannequin, the tracker hands out a fresh id each time, and
a per-track timer would restart forever (the store's entrance dummy sat at
track #231 counting as a visitor for exactly this reason). Still-seconds
accumulate on the quantized foot cell, a successor track standing on a cell
whose bank has crossed the threshold latches right away, and a track that
genuinely WALKS away clears its cell's bank so checkout queues can't
accumulate one shopper at a time into a phantom mannequin.
"""

from __future__ import annotations

import math
import time


class MannequinFilter:
    def __init__(
        self,
        still_after_sec: float = 180.0,
        move_frac: float = 0.03,
        spot_cell: float = 0.05,
        spot_ttl_sec: float = 900.0,
    ) -> None:
        # A track must stay within `move_frac` (of frame size, normalized coords)
        # of its anchor for `still_after_sec` before the stillness rule latches.
        self.still_after_sec = still_after_sec
        self.move_frac = move_frac
        # Spot bank: still-seconds accumulated per quantized foot cell. Cell
        # size is deliberately larger than move_frac so a successor track's
        # anchor lands in the same or an adjacent cell despite jitter.
        self.spot_cell = spot_cell
        self.spot_ttl_sec = spot_ttl_sec
        self._anchor: dict[int, tuple[float, float, float]] = {}  # tid → (x, y, since)
        self._latched: set[int] = set()
        self._last_seen: dict[int, float] = {}
        self._spot_still: dict[tuple[int, int], list[float]] = {}  # cell → [accum, last_ts]
        self._bank_cell: dict[int, tuple[int, int]] = {}  # tid → cell it banks into

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x / self.spot_cell), int(y / self.spot_cell))

    def _richest_cell(self, cell: tuple[int, int]) -> tuple[int, int]:
        """The best-funded cell in the 3×3 neighborhood (own cell when none) —
        churn jitter lands a successor's anchor one cell over, and its banking
        must CONTINUE that bank, not start a parallel one that never combines."""
        best, best_acc = cell, -1.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = (cell[0] + dx, cell[1] + dy)
                rec = self._spot_still.get(c)
                acc = rec[0] if rec else 0.0
                if acc > best_acc or (acc == best_acc and c == cell):
                    best, best_acc = c, acc
        return best

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
        prev_seen = self._last_seen.get(tid)
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
                # It walks → it's no mannequin (anymore). Release, re-anchor,
                # and void the spot's bank — the spot was clearly occupiable.
                self._latched.discard(tid)
                self._spot_still.pop(self._bank_cell.pop(tid, self._cell(ax, ay)), None)
                self._anchor[tid] = (foot_nx, foot_ny, now)
                return False
            return True

        if moved:
            # A real person stood here and left — the bank must not carry
            # over to the NEXT shopper pausing on the same tile.
            self._spot_still.pop(self._bank_cell.pop(tid, self._cell(ax, ay)), None)
            self._anchor[tid] = (foot_nx, foot_ny, now)
            return False

        # Bank this frame's stillness on the neighborhood's richest cell (a
        # churned predecessor's bank continues instead of forking). dt is
        # capped so a detector blinking for seconds isn't credited for the gap.
        cell = self._bank_cell.get(tid) or self._richest_cell(self._cell(ax, ay))
        self._bank_cell[tid] = cell
        if prev_seen is not None:
            rec = self._spot_still.setdefault(cell, [0.0, now])
            rec[0] += min(now - prev_seen, 2.0)
            rec[1] = now

        # Rule 2: still past the threshold — on THIS track, or cumulatively on
        # this spot across earlier (churned) track ids → latch.
        bank = self._spot_still.get(cell)
        if now - since >= self.still_after_sec or (bank and bank[0] >= self.still_after_sec):
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
            self._bank_cell.pop(t, None)
        # Spot banks decay too — an unreinforced cell (dummy removed, or noise)
        # must not latch somebody standing there half an hour later.
        dead = [c for c, rec in self._spot_still.items() if now - rec[1] > self.spot_ttl_sec]
        for c in dead:
            self._spot_still.pop(c, None)
