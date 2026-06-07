"""Cross-camera re-ID registry + histogram embedder (ADR-0022/0023)."""

from __future__ import annotations

import numpy as np

from sentry_ai.live_worker.reid import (
    HistogramEmbedder,
    StorePersonRegistry,
    cosine,
)


def _solid(color: tuple[int, int, int], h: int = 40, w: int = 20) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_embedder_normalized_and_distinguishes_colors() -> None:
    emb = HistogramEmbedder(bins=8)
    red = emb.embed(_solid((0, 0, 200)), (0, 0, 20, 40))
    blue = emb.embed(_solid((200, 0, 0)), (0, 0, 20, 40))
    assert red is not None and blue is not None
    assert abs(float(np.linalg.norm(red)) - 1.0) < 1e-5
    # Different dominant colors → low similarity.
    assert cosine(red, blue) < 0.5


def test_embedder_same_color_high_similarity() -> None:
    emb = HistogramEmbedder(bins=8)
    a = emb.embed(_solid((0, 180, 0)), (0, 0, 20, 40))
    b = emb.embed(_solid((0, 180, 0)), (0, 0, 20, 40))
    assert a is not None and b is not None
    assert cosine(a, b) > 0.99


def test_embedder_handles_empty_box() -> None:
    emb = HistogramEmbedder()
    assert emb.embed(_solid((1, 1, 1)), (5, 5, 5, 5)) is None  # zero-area
    assert emb.embed(_solid((1, 1, 1)), (-10, -10, -1, -1)) is None


def test_registry_matches_same_person_across_cameras() -> None:
    reg = StorePersonRegistry(match_threshold=0.6)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 20, 40))
    assert v is not None
    # Seen on cam A, then again (same look) on cam B → SAME store-person id.
    id_a = reg.match_or_create(v, "camA", now=1000.0)
    id_b = reg.match_or_create(v, "camB", now=1001.0)
    assert id_a == id_b
    assert reg.camera_count(id_a) == 2
    assert len(reg) == 1


def test_registry_separates_distinct_people() -> None:
    reg = StorePersonRegistry(match_threshold=0.6)
    emb = HistogramEmbedder(bins=8)
    red = emb.embed(_solid((0, 0, 200)), (0, 0, 20, 40))
    blue = emb.embed(_solid((200, 0, 0)), (0, 0, 20, 40))
    assert red is not None and blue is not None
    id1 = reg.match_or_create(red, "camA", now=1000.0)
    id2 = reg.match_or_create(blue, "camA", now=1000.0)
    assert id1 != id2
    assert len(reg) == 2


def test_registry_accumulates_score_across_cameras() -> None:
    reg = StorePersonRegistry(match_threshold=0.6)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 20, 40))
    assert v is not None
    pid = reg.match_or_create(v, "camA", now=1000.0)
    reg.add_score(pid, 20.0, now=1000.0)
    # Same person reappears on cam B and adds more — score accumulates.
    reg.match_or_create(v, "camB", now=1001.0)
    total = reg.add_score(pid, 15.0, now=1001.0)
    assert total == 35.0


def test_registry_prunes_outside_window() -> None:
    reg = StorePersonRegistry(match_threshold=0.6, window_sec=60.0)
    emb = HistogramEmbedder(bins=8)
    v = emb.embed(_solid((10, 200, 30)), (0, 0, 20, 40))
    assert v is not None
    reg.match_or_create(v, "camA", now=1000.0)
    # A later match past the window prunes the stale person first → new id, len 1.
    reg.match_or_create(v, "camA", now=2000.0)
    assert len(reg) == 1
