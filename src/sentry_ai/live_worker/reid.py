"""Cross-camera person re-identification + per-store score accumulation.

ADR-0022 (accumulate one person's risk across cameras) + ADR-0023 (a store's
cameras all run on ONE node → the registry is an in-process, store-scoped
structure; no central DB needed). It matches appearance embeddings across a
store's cameras within a time window and accumulates each person's risk so a
thief building suspicion across aisles is caught at any camera.

The `Embedder` is pluggable. The default `HistogramEmbedder` is dependency-light
(numpy only) but WEAK — people in similar clothing collide. Swap in a learned
re-ID model (OSNet / torchreid) for production accuracy; that is where labeled
tuning + GPU budget come in. `StorePersonRegistry` is deliberately a small API so
a shared backend (Redis / pgvector) can replace the in-memory impl for the rare
store too large for a single GPU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.live_worker.reid")

Box = tuple[float, float, float, float]


class Embedder(Protocol):
    """Produces an L2-normalized appearance vector for a person crop."""

    def embed(self, frame_bgr: NDArray[np.uint8], box: Box) -> NDArray[np.float32] | None: ...


class HistogramEmbedder:
    """Per-channel color-histogram appearance embedding (numpy-only, L2-norm).

    A dependency-light v1: good enough to re-link a person with distinctive
    clothing across adjacent cameras, but NOT robust (similar outfits collide).
    Replace with a learned re-ID model for production.
    """

    def __init__(self, bins: int = 8) -> None:
        self.bins = bins

    def embed(self, frame_bgr: NDArray[np.uint8], box: Box) -> NDArray[np.float32] | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2].reshape(-1, 3)
        if crop.size == 0:
            return None
        edges = np.linspace(0, 256, self.bins + 1)
        parts = [np.histogram(crop[:, c], bins=edges)[0] for c in range(3)]
        vec = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return None
        return vec / norm


class OSNetEmbedder:
    """Learned re-ID embedding via torchreid's OSNet (#4).

    Far more robust than the color histogram — distinguishes people in similar
    clothing. Requires the optional `torchreid` + `torch` deps and (ideally) a
    GPU. Construction raises if torchreid is unavailable, so `make_embedder`
    falls back to the histogram. Produces an L2-normalized feature vector.
    """

    def __init__(self, model_name: str = "osnet_x0_25", device: str | None = None) -> None:
        # Lazy heavy imports — raise so the factory can fall back gracefully.
        import torch  # noqa: PLC0415
        from torchreid.utils import FeatureExtractor  # noqa: PLC0415

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._extractor = FeatureExtractor(model_name=model_name, device=dev)
        log.info("reid.osnet_loaded", model=model_name, device=dev)

    def embed(self, frame_bgr: NDArray[np.uint8], box: Box) -> NDArray[np.float32] | None:
        import cv2  # noqa: PLC0415

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # torchreid expects RGB HWC; FeatureExtractor accepts a list of ndarrays.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        raw = self._extractor([rgb]).cpu().numpy()[0]
        feats: NDArray[np.float32] = np.asarray(raw, dtype=np.float32)
        norm = float(np.linalg.norm(feats))
        if norm == 0.0:
            return None
        return feats / norm


def make_embedder(name: str = "histogram") -> Embedder:
    """Resolve a re-ID embedder by name, falling back to the histogram if the
    requested learned model can't be loaded (missing deps / no GPU)."""
    key = (name or "histogram").lower()
    if key in ("histogram", "hist", ""):
        return HistogramEmbedder()
    if key in ("osnet", "torchreid"):
        try:
            return OSNetEmbedder()
        except Exception as e:  # noqa: BLE001 — any import/load failure → fallback
            log.warning("reid.osnet_unavailable_fallback_histogram", error=str(e))
            return HistogramEmbedder()
    log.warning("reid.unknown_embedder_fallback_histogram", requested=key)
    return HistogramEmbedder()


def cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity of two same-length vectors (0 if either is degenerate)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass(slots=True)
class StorePerson:
    person_id: int
    embedding: NDArray[np.float32]
    score: float = 0.0
    last_seen: float = field(default_factory=time.time)
    cameras: set[str] = field(default_factory=set)


class StorePersonRegistry:
    """In-process, store-scoped re-ID + score accumulation. Thread-safe.

    One instance per store (the node holds a dict[store_id → registry]).
    """

    def __init__(
        self,
        match_threshold: float = 0.6,
        window_sec: float = 1800.0,
        ema: float = 0.9,
    ) -> None:
        self.match_threshold = match_threshold
        self.window_sec = window_sec
        self.ema = ema
        self._people: dict[int, StorePerson] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def match_or_create(
        self, embedding: NDArray[np.float32], camera_id: str, now: float | None = None
    ) -> int:
        """Return the store-global person id for this embedding, creating one if no
        existing person within the time window is similar enough."""
        ts = time.time() if now is None else now
        with self._lock:
            self._prune(ts)
            best_id = -1
            best_sim = self.match_threshold
            for pid, p in self._people.items():
                sim = cosine(embedding, p.embedding)
                if sim >= best_sim:
                    best_sim = sim
                    best_id = pid
            if best_id != -1:
                p = self._people[best_id]
                # EMA-update the appearance so it tracks lighting/pose drift.
                p.embedding = (self.ema * p.embedding + (1.0 - self.ema) * embedding).astype(
                    np.float32
                )
                p.last_seen = ts
                p.cameras.add(camera_id)
                return best_id
            pid = self._next_id
            self._next_id += 1
            self._people[pid] = StorePerson(
                person_id=pid,
                embedding=embedding.astype(np.float32),
                last_seen=ts,
                cameras={camera_id},
            )
            return pid

    def add_score(self, person_id: int, delta: float, now: float | None = None) -> float:
        """Add to a person's accumulated cross-camera score; return the new total."""
        ts = time.time() if now is None else now
        with self._lock:
            p = self._people.get(person_id)
            if p is None:
                return 0.0
            p.score += delta
            p.last_seen = ts
            return p.score

    def get_score(self, person_id: int) -> float:
        with self._lock:
            p = self._people.get(person_id)
            return p.score if p else 0.0

    def camera_count(self, person_id: int) -> int:
        with self._lock:
            p = self._people.get(person_id)
            return len(p.cameras) if p else 0

    def prune(self, now: float | None = None) -> int:
        ts = time.time() if now is None else now
        with self._lock:
            return self._prune(ts)

    def _prune(self, ts: float) -> int:
        stale = [pid for pid, p in self._people.items() if ts - p.last_seen > self.window_sec]
        for pid in stale:
            del self._people[pid]
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._people)
