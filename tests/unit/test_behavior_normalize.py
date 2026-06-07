"""ADR-0022 scoring model: raw→0-100 normalization + loitering dimension."""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.behavior import (
    DEFAULT_GREEN_MAX,
    DEFAULT_RED_MAX,
    DEFAULT_YELLOW_MAX,
    BehaviorScorer,
    normalize_pct,
)

L_EYE, R_EYE = 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def test_normalize_band_anchors() -> None:
    g, y, r = DEFAULT_GREEN_MAX, DEFAULT_YELLOW_MAX, DEFAULT_RED_MAX
    assert normalize_pct(0, g, y, r) == 0.0
    assert normalize_pct(g, g, y, r) == 30.0  # green edge → 30%
    assert normalize_pct(y, g, y, r) == 70.0  # yellow edge (breach) → 70%
    assert normalize_pct(r, g, y, r) == 100.0  # red anchor → 100%
    assert normalize_pct(r * 5, g, y, r) == 100.0  # capped


def test_normalize_monotonic_and_bounded() -> None:
    g, y, r = DEFAULT_GREEN_MAX, DEFAULT_YELLOW_MAX, DEFAULT_RED_MAX
    prev = -1.0
    for raw in range(60):
        pct = normalize_pct(float(raw), g, y, r)
        assert 0.0 <= pct <= 100.0
        assert pct >= prev  # non-decreasing
        prev = pct


def test_midband_values() -> None:
    # halfway through the yellow band (raw=10, between 5 and 15) → 50%
    assert normalize_pct(10.0, 5.0, 15.0, 30.0) == 50.0


def _person_at(cx: float, cy_hip: float) -> np.ndarray:
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[L_SHOULDER] = (cx - 10, 100)
    kps[R_SHOULDER] = (cx + 10, 100)
    kps[L_EYE] = (cx - 4, 60)
    kps[R_EYE] = (cx + 4, 60)
    kps[L_HIP] = (cx - 8, cy_hip)
    kps[R_HIP] = (cx + 8, cy_hip)
    return kps


def test_loitering_scores_after_dwell(monkeypatch) -> None:
    import sentry_ai.live_worker.behavior as bx

    scorer = BehaviorScorer(weights={"loitering": 5.0})
    kps = _person_at(100.0, 300.0)
    person_h = 200.0

    t = [1000.0]
    monkeypatch.setattr(bx.time, "time", lambda: t[0])

    # First frame anchors the dwell — no loiter score yet.
    s0, _, _ = scorer.score(1, kps, person_h)
    # Advance past the loiter window while staying put → loiter fires.
    t[0] += bx.LOITER_SECONDS + 1
    _, _, reasons = scorer.score(1, kps, person_h)
    assert any("Удаан зогсох" in r for r in reasons)


def test_moving_person_does_not_loiter(monkeypatch) -> None:
    import sentry_ai.live_worker.behavior as bx

    scorer = BehaviorScorer(weights={"loitering": 5.0})
    person_h = 200.0
    t = [1000.0]
    monkeypatch.setattr(bx.time, "time", lambda: t[0])

    scorer.score(1, _person_at(100.0, 300.0), person_h)
    t[0] += bx.LOITER_SECONDS + 1
    # Moved far → anchor resets, no loiter.
    _, _, reasons = scorer.score(1, _person_at(900.0, 300.0), person_h)
    assert not any("Удаан зогсох" in r for r in reasons)
