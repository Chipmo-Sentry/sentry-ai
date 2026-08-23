"""_PoseBatcher (yolo_runner): concurrent cameras' frames must merge into one
batched predict, each caller must get ITS OWN frame's result back, and a
predict failure must reach every waiting caller instead of hanging them."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from sentry_ai.live_worker import yolo_runner
from sentry_ai.live_worker.yolo_runner import _PoseBatcher


class _FakeModel:
    """predict() echoes each frame's marker value so callers can verify the
    positional mapping; records call sizes to prove batching happened."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.lock = threading.Lock()

    def predict(self, frames, **_kw):  # type: ignore[no-untyped-def]
        with self.lock:
            self.calls.append(len(frames))
        return [("res", int(f[0, 0, 0])) for f in frames]


@pytest.fixture()
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeModel:
    m = _FakeModel()
    monkeypatch.setattr(yolo_runner, "_MODEL", m)
    monkeypatch.setattr(yolo_runner, "_DEVICE", "cpu")
    return m


def _frame(marker: int) -> np.ndarray:
    return np.full((8, 8, 3), marker, dtype=np.uint8)


def test_concurrent_submits_batch_and_map_back(fake_model: _FakeModel) -> None:
    b = _PoseBatcher(window_ms=200, max_batch=8)
    results: dict[int, object] = {}

    def worker(marker: int) -> None:
        results[marker] = b.infer(_frame(marker))

    threads = [threading.Thread(target=worker, args=(m,)) for m in (10, 20, 30, 40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    # Every caller got exactly its own frame's echo back.
    assert results == {m: ("res", m) for m in (10, 20, 30, 40)}
    # The 200ms window merged the concurrent submits into few real GPU calls —
    # allow ≤2 for scheduling jitter, but 4 separate calls would mean no batching.
    assert sum(fake_model.calls) == 4
    assert len(fake_model.calls) <= 2


def test_max_batch_caps_one_call(fake_model: _FakeModel) -> None:
    b = _PoseBatcher(window_ms=200, max_batch=2)
    results: dict[int, object] = {}
    threads = [
        threading.Thread(target=lambda m=m: results.__setitem__(m, b.infer(_frame(m))))
        for m in (1, 2, 3, 4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert results == {m: ("res", m) for m in (1, 2, 3, 4)}
    assert max(fake_model.calls) <= 2  # cap respected


def test_predict_failure_reaches_all_callers(
    fake_model: _FakeModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(frames, **_kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("CUDA fell over")

    monkeypatch.setattr(fake_model, "predict", boom)
    b = _PoseBatcher(window_ms=50, max_batch=4)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            b.infer(_frame(1))
        except RuntimeError as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert len(errors) == 3  # nobody hangs; everyone sees the failure

    # And the collector survives to serve the next (healthy) batch.
    monkeypatch.setattr(
        fake_model, "predict", lambda frames, **_kw: [("res", int(f[0, 0, 0])) for f in frames]
    )
    assert b.infer(_frame(7)) == ("res", 7)
