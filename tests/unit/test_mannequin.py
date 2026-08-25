"""Mannequin filter: zone rule latches instantly, stillness latches after the
threshold, and BOTH release the moment the track genuinely moves."""

from __future__ import annotations

from sentry_ai.live_worker.mannequin import MannequinFilter


def test_mannequin_zone_latches_immediately() -> None:
    f = MannequinFilter()
    assert f.observe(1, 0.5, 0.5, {"mannequin"}, now=0.0) is True
    # Still flagged on later frames even outside the zone set (e.g. zones=None
    # frame glitch) while it hasn't moved.
    assert f.observe(1, 0.5, 0.5, None, now=1.0) is True


def test_stillness_latches_after_threshold_and_releases_on_move() -> None:
    f = MannequinFilter(still_after_sec=60.0, move_frac=0.03)
    assert f.observe(2, 0.4, 0.6, None, now=0.0) is False
    assert f.observe(2, 0.405, 0.6, None, now=30.0) is False  # sub-threshold sway
    assert f.observe(2, 0.4, 0.6, None, now=61.0) is True  # still past 60s → latch
    assert f.observe(2, 0.41, 0.6, None, now=62.0) is True  # micro-sway stays latched
    assert f.observe(2, 0.6, 0.6, None, now=63.0) is False  # it WALKS → release
    assert f.observe(2, 0.6, 0.6, None, now=64.0) is False


def test_moving_person_never_latches() -> None:
    f = MannequinFilter(still_after_sec=60.0, move_frac=0.03)
    for i in range(20):
        # Strolls across the frame — anchor re-sets every step.
        assert f.observe(3, 0.05 * i, 0.5, {"shelf"}, now=float(i * 10)) is False


def test_prune_forgets_stale_tracks() -> None:
    f = MannequinFilter(still_after_sec=10.0)
    f.observe(4, 0.5, 0.5, {"mannequin"}, now=0.0)
    f.prune(ttl_sec=5.0, now=100.0)
    # Track id recycled by the tracker later — starts fresh, not pre-latched.
    assert f.observe(4, 0.2, 0.2, None, now=101.0) is False
