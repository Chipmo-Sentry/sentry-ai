"""Demographics vote cache (docs/30 F5): banding, stability gating, cadence,
per-frame cost cap, lock, and prune — all with a fake classifier (no models,
no network). The backend counts each (camera, person_id) once with FIRST
values winning, so the None-until-stable behaviour here is load-bearing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sentry_ai.live_worker.demographics import (
    Box,
    FaceVote,
    TrackDemographics,
    band_probs_from_age,
)
from sentry_ai.live_worker.schemas import TrackPayload

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)

MALE_ADULT = FaceVote(gender_probs=(0.9, 0.1), band_probs=(0.0, 0.1, 0.8, 0.1), face_score=0.9)
FEMALE_YOUTH = FaceVote(gender_probs=(0.2, 0.8), band_probs=(0.1, 0.7, 0.2, 0.0), face_score=0.9)


class FakeTrack:
    def __init__(self, tid: int) -> None:
        self.tracker_id = tid
        self.box: Box = (10.0, 10.0, 110.0, 210.0)


class FakeClassifier:
    """Returns a scripted sequence of votes (cycles the last one), counts calls."""

    def __init__(self, *votes: FaceVote | None) -> None:
        self._votes = list(votes)
        self.calls = 0

    def classify(self, frame_bgr: NDArray[np.uint8], box: Box) -> FaceVote | None:
        self.calls += 1
        if not self._votes:
            return None
        return self._votes.pop(0) if len(self._votes) > 1 else self._votes[0]


def test_band_probs_from_age_maps_buckets_to_bands() -> None:
    child = band_probs_from_age(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert child == (1.0, 0.0, 0.0, 0.0)
    senior = band_probs_from_age(np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    assert senior == (0.0, 0.0, 0.0, 1.0)
    # (4-6) + (15-20) + (25-32) split across child/youth/adult
    mixed = band_probs_from_age(np.array([0, 0.5, 0, 0.3, 0.2, 0, 0, 0], dtype=np.float32))
    assert mixed[0] == 0.5 and abs(mixed[1] - 0.3) < 1e-6 and abs(mixed[2] - 0.2) < 1e-6


def test_labels_none_until_min_votes_then_stable() -> None:
    td = TrackDemographics(FakeClassifier(MALE_ADULT), every_n=1, min_votes=2, max_votes=5)
    tracks = [FakeTrack(1)]
    assert td.observe(FRAME, tracks, 0)[1] == (None, None)  # 1 vote < min_votes
    assert td.observe(FRAME, tracks, 1)[1] == ("male", "adult")


def test_weighted_votes_resolve_disagreement() -> None:
    # Strong male vote + weaker female vote → male wins on probability mass.
    weak_female = FaceVote(gender_probs=(0.4, 0.6), band_probs=(0.0, 0.0, 1.0, 0.0), face_score=0.8)
    td = TrackDemographics(
        FakeClassifier(MALE_ADULT, weak_female), every_n=1, min_votes=2, max_votes=5
    )
    tracks = [FakeTrack(1)]
    td.observe(FRAME, tracks, 0)
    assert td.observe(FRAME, tracks, 1)[1] == ("male", "adult")


def test_cadence_gates_attempts_per_track() -> None:
    clf = FakeClassifier(MALE_ADULT)
    td = TrackDemographics(clf, every_n=5, min_votes=2, max_votes=99)
    tracks = [FakeTrack(1)]
    for idx in range(5):
        td.observe(FRAME, tracks, idx)
    assert clf.calls == 1  # frame 0 only
    td.observe(FRAME, tracks, 5)
    assert clf.calls == 2  # next attempt exactly every_n later


def test_lock_after_max_votes_stops_classifying() -> None:
    clf = FakeClassifier(MALE_ADULT)
    td = TrackDemographics(clf, every_n=1, min_votes=1, max_votes=2)
    tracks = [FakeTrack(1)]
    for idx in range(10):
        assert td.observe(FRAME, tracks, idx)[1] == ("male", "adult") or idx == 0
    assert clf.calls == 2  # locked after max_votes — attribute, not per-frame state


def test_failed_detection_keeps_labels_none_but_costs_an_attempt() -> None:
    clf = FakeClassifier(None)
    td = TrackDemographics(clf, every_n=1, min_votes=1, max_votes=3)
    tracks = [FakeTrack(1)]
    for idx in range(4):
        assert td.observe(FRAME, tracks, idx)[1] == (None, None)
    assert clf.calls == 4  # no face → keeps retrying (no votes banked)


def test_max_per_frame_caps_cost_and_round_robins() -> None:
    clf = FakeClassifier(MALE_ADULT)
    td = TrackDemographics(clf, every_n=1, min_votes=1, max_votes=1, max_per_frame=1)
    tracks = [FakeTrack(1), FakeTrack(2), FakeTrack(3)]
    td.observe(FRAME, tracks, 0)
    assert clf.calls == 1
    td.observe(FRAME, tracks, 1)
    td.observe(FRAME, tracks, 2)
    assert clf.calls == 3
    # All three locked after one vote each (max_votes=1) → all labeled.
    out = td.observe(FRAME, tracks, 3)
    assert clf.calls == 3
    assert all(out[tid] == ("male", "adult") for tid in (1, 2, 3))


def test_prune_drops_idle_track_state() -> None:
    import time

    td = TrackDemographics(FakeClassifier(MALE_ADULT), every_n=1, min_votes=1, max_votes=1)
    td.observe(FRAME, [FakeTrack(1)], 0)
    assert len(td._states) == 1
    td.prune(now=time.monotonic() + 120.0)
    assert len(td._states) == 0


def test_track_payload_demographics_default_none_and_accepts_values() -> None:
    p = TrackPayload(person_id=1, box=(0, 0, 10, 10), det_confidence=0.9)
    assert p.gender is None and p.age_band is None
    dumped = p.model_dump(mode="json")
    assert dumped["gender"] is None and dumped["age_band"] is None
    q = TrackPayload(
        person_id=2, box=(0, 0, 10, 10), det_confidence=0.9, gender="female", age_band="youth"
    )
    assert q.gender == "female" and q.age_band == "youth"
