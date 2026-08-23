"""Motion gate (live_worker/motion_gate.py): YOLO must be skipped ONLY when the
scene is simultaneously still and track-free, reopen instantly on motion or an
active track, and keep the idle safety-net cadence."""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.motion_gate import MotionGate


def _still() -> np.ndarray:
    return np.full((360, 640, 3), 80, dtype=np.uint8)


def _moving(seed: int) -> np.ndarray:
    f = _still()
    # A person-sized bright blob whose position depends on the seed → large diff.
    x = 50 + (seed * 40) % 400
    f[100:300, x : x + 80] = 220
    return f


def test_closes_only_after_quiet_streak_then_keeps_safety_cadence() -> None:
    g = MotionGate(quiet_after=5, idle_stride=4)
    assert g.observe(_still(), active_tracks=0).infer  # first frame = motion (no prev)
    for _ in range(4):
        assert g.observe(_still(), active_tracks=0).infer  # warm-down keeps inferring
    decisions = [g.observe(_still(), active_tracks=0) for _ in range(12)]
    assert all(d.idle for d in decisions)
    ran = [d.infer for d in decisions]
    # Every idle_stride-th frame still runs YOLO — never a total blackout.
    assert ran.count(True) == 3 and g.gated_total == 9


def test_motion_reopens_immediately() -> None:
    g = MotionGate(quiet_after=3, idle_stride=4)
    for _ in range(10):
        g.observe(_still(), active_tracks=0)
    d = g.observe(_moving(1), active_tracks=0)
    assert d.infer and not d.idle  # the frame someone enters IS analyzed


def test_active_track_holds_gate_open_when_scene_still() -> None:
    # A person standing motionless produces no diff — their live track must keep
    # full-rate detection anyway.
    g = MotionGate(quiet_after=3, idle_stride=4)
    for _ in range(20):
        d = g.observe(_still(), active_tracks=1)
        assert d.infer and not d.idle
    assert g.gated_total == 0


def test_disabled_gate_always_infers() -> None:
    g = MotionGate(enabled=False, quiet_after=1, idle_stride=2)
    for _ in range(10):
        assert g.observe(_still(), active_tracks=0).infer
    assert g.gated_total == 0
