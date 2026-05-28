"""ByteTrack wrapper over supervision lib.

Assigns persistent `tracker_id` to detections across frames so we can build
per-person tracks for risk scoring (L4) and threshold-breach detection (L5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sentry_ai.live_worker.yolo_runner import Detection


@dataclass(slots=True)
class TrackedDetection:
    box: tuple[float, float, float, float]
    score: float
    tracker_id: int


class ByteTrackWrapper:
    """Wraps `supervision.ByteTrack` with our Detection / TrackedDetection types.

    ByteTrack maintains in-memory state per instance — DO NOT share across cameras.
    Each `CameraWorker` instantiates its own.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 10,
    ) -> None:
        # Lazy import — supervision pulls heavy deps
        import supervision as sv

        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self._sv = sv

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        if not detections:
            # Still tick the tracker so lost-track buffer ages correctly
            self._tracker.update_with_detections(self._sv.Detections.empty())
            return []

        xyxy: NDArray[np.float32] = np.array([d.box for d in detections], dtype=np.float32)
        conf: NDArray[np.float32] = np.array([d.score for d in detections], dtype=np.float32)
        # Class id required by supervision — all persons → 0
        class_id: NDArray[np.int32] = np.zeros(len(detections), dtype=np.int32)

        sv_dets = self._sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
        tracked = self._tracker.update_with_detections(sv_dets)

        out: list[TrackedDetection] = []
        if tracked.tracker_id is None:
            return out
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i]
            if tid is None or tid < 0:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i].tolist()
            score = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            out.append(
                TrackedDetection(
                    box=(x1, y1, x2, y2),
                    score=score,
                    tracker_id=int(tid),
                ),
            )
        return out
