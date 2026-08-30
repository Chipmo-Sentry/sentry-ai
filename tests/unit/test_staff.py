"""Staff badge vote cache (owner plan 2026-07-17): color parsing, chest-region
color matching, vote-lock (color-only path), re-ID staff latch, config gating,
and prune — no VLM, no network (vlm_verify=False everywhere)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from sentry_ai import runtime_config
from sentry_ai.live_worker.reid import StorePersonRegistry
from sentry_ai.live_worker.staff import (
    Box,
    TrackStaff,
    _chest_box,
    _color_frac,
    parse_badge_color,
)
from sentry_ai.live_worker.schemas import TrackPayload


class FakeTrack:
    def __init__(self, tid: int, box: Box = (40.0, 20.0, 200.0, 400.0)) -> None:
        self.tracker_id = tid
        self.box = box
        self.keypoints: NDArray[np.float32] | None = None


def _frame_with_chest_patch(bgr: tuple[int, int, int] | None) -> NDArray[np.uint8]:
    """A 480x320 gray frame; when `bgr` is set, the fallback chest region of the
    FakeTrack default box gets a solid patch of that color."""
    frame = np.full((480, 320, 3), 128, dtype=np.uint8)
    if bgr is not None:
        # FakeTrack box (40,20,200,400) → fallback chest: x 72..168, y 88..191.
        frame[90:190, 75:165] = bgr
    return frame


@pytest.fixture(autouse=True)
def _badge_color():
    """Each test starts with the central badge color set to orange, and leaves
    the process-global runtime_config clean afterwards."""
    runtime_config.set_staff_badge_color("orange")
    yield
    runtime_config.set_staff_badge_color(None)


def _make(**kw) -> TrackStaff:
    kw.setdefault("every_n", 1)
    kw.setdefault("max_per_frame", 8)
    kw.setdefault("min_hits", 2)
    kw.setdefault("color_only_hits", 3)
    kw.setdefault("vlm_verify", False)
    return TrackStaff(**kw)


def test_parse_badge_color_named_and_hex() -> None:
    assert parse_badge_color("orange") is not None
    assert parse_badge_color("#ff6a00") is not None
    assert parse_badge_color("#808080") is None  # gray: unmatchable, rejected
    assert parse_badge_color("bogus") is None


def test_chest_box_uses_shoulders_when_confident() -> None:
    box: Box = (0.0, 0.0, 100.0, 300.0)
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[5] = (30.0, 80.0, 0.9)  # l_shoulder
    kps[6] = (70.0, 80.0, 0.9)  # r_shoulder
    cx1, cy1, cx2, cy2 = _chest_box(box, kps)
    assert 30.0 < cx1 < cx2 < 70.0
    assert cy1 == pytest.approx(80.0)
    assert cy2 > cy1


def test_color_frac_detects_patch() -> None:
    hsv = parse_badge_color("orange")
    assert hsv is not None
    frame = _frame_with_chest_patch((0, 106, 255))  # orange in BGR
    chest = _chest_box((40.0, 20.0, 200.0, 400.0), None)
    assert _color_frac(frame, chest, hsv) > 0.5
    assert _color_frac(_frame_with_chest_patch(None), chest, hsv) < 0.01


def test_color_only_lock_after_enough_hits() -> None:
    ts = _make()
    t = FakeTrack(1)
    frame = _frame_with_chest_patch((0, 106, 255))
    out = {}
    for i in range(4):
        out = ts.observe(frame, [t], i)
    assert out[1] is True
    # Locked stays locked — a later colorless frame doesn't flip it.
    out = ts.observe(_frame_with_chest_patch(None), [t], 99)
    assert out[1] is True


def test_visitor_without_badge_never_staff() -> None:
    ts = _make()
    t = FakeTrack(2)
    frame = _frame_with_chest_patch(None)
    out = {}
    for i in range(10):
        out = ts.observe(frame, [t], i)
    assert out[2] is False


def test_no_color_configured_is_inert() -> None:
    runtime_config.set_staff_badge_color(None)
    ts = _make()
    t = FakeTrack(3)
    out = ts.observe(_frame_with_chest_patch((0, 106, 255)), [t], 0)
    assert out[3] is False


def test_color_change_resets_votes() -> None:
    ts = _make()
    t = FakeTrack(4)
    frame = _frame_with_chest_patch((0, 106, 255))
    ts.observe(frame, [t], 0)
    ts.observe(frame, [t], 1)
    runtime_config.set_staff_badge_color("blue")
    out = ts.observe(frame, [t], 2)  # orange patch no longer matches blue
    assert out[4] is False


def test_store_color_overrides_node_global() -> None:
    """A per-store color wins over the node-global one: the node says orange,
    but this store's workers use blue, so a blue lanyard locks staff."""
    ts = _make(store_color="blue")  # node-global is "orange" (fixture)
    t = FakeTrack(11)
    frame = _frame_with_chest_patch((255, 0, 0))  # blue in BGR
    out = {}
    for i in range(4):
        out = ts.observe(frame, [t], i)
    assert out[11] is True


def test_store_color_ignores_node_global_color() -> None:
    """With a store color set, the node-global orange is NOT matched — an orange
    lanyard is a visitor at a store whose staff wear blue."""
    ts = _make(store_color="blue")
    t = FakeTrack(12)
    frame = _frame_with_chest_patch((0, 106, 255))  # orange in BGR
    out = {}
    for i in range(10):
        out = ts.observe(frame, [t], i)
    assert out[12] is False


def test_no_store_color_falls_back_to_node_global() -> None:
    """Without a store color, the node-global color still applies (single-store
    deployments keep working unchanged)."""
    ts = _make(store_color=None)  # node-global "orange" from the fixture
    t = FakeTrack(13)
    frame = _frame_with_chest_patch((0, 106, 255))  # orange
    out = {}
    for i in range(4):
        out = ts.observe(frame, [t], i)
    assert out[13] is True


def test_registry_staff_latch() -> None:
    reg = StorePersonRegistry()
    emb = np.ones(8, dtype=np.float32)
    pid = reg.match_or_create(emb, "cam1")
    assert reg.is_staff(pid) is False
    reg.mark_staff(pid)
    assert reg.is_staff(pid) is True
    assert reg.is_staff(999) is False


def test_track_payload_default_not_staff() -> None:
    p = TrackPayload(person_id=1, box=(0, 0, 1, 1), det_confidence=0.5)
    assert p.is_staff is False
