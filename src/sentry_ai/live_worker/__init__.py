"""Per-camera live worker — RTSP pull + YOLO + tracker + metadata emit.

M1-LIVE Phase L2.

Architecture:
  CameraWorker (one thread per camera)
    cv2.VideoCapture(rtsp_url) → frame
      └─ every Nth frame ─→ YoloRunner.detect(frame)
                              └─ ByteTrackWrapper.update(detections)
                                  └─ MetadataEmitter.enqueue(payload)

  Manager
    spawns + supervises CameraWorker threads
    exposes start/stop/status to FastAPI router

  MetadataEmitter
    single background thread drains a queue and POSTs to sentry-backend
    `POST /api/v1/internal/live-metadata` (batched, drop-oldest on overflow)
"""

from sentry_ai.live_worker.manager import LiveWorkerManager, get_manager

__all__ = ["LiveWorkerManager", "get_manager"]
