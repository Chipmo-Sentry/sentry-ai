"""Metadata emitter — drains a thread-safe queue and POSTs to sentry-backend.

One background thread per process. Queue is drop-oldest on overflow (live > complete).
"""

from __future__ import annotations

import queue
import threading
import time

import httpx

from sentry_ai.live_worker.schemas import FrameMetadata
from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.live_worker.emitter")

# Queue cap — at 10 FPS per cam × 2 cams = 20 items/sec; 300 = ~15 sec buffer
_QUEUE_CAP = 300
_BATCH_SIZE = 20
_BATCH_TIMEOUT_SEC = 0.2
_REQUEST_TIMEOUT_SEC = 5.0


class MetadataEmitter:
    """Singleton-ish — created by manager, accessed by all camera workers."""

    def __init__(self) -> None:
        self._queue: queue.Queue[FrameMetadata] = queue.Queue(maxsize=_QUEUE_CAP)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._dropped = 0
        self._sent = 0
        self._failed = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="live-metadata-emitter",
            daemon=True,
        )
        self._thread.start()
        log.info("emitter.started", queue_cap=_QUEUE_CAP, batch_size=_BATCH_SIZE)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        log.info("emitter.stopped", sent=self._sent, dropped=self._dropped, failed=self._failed)

    def enqueue(self, metadata: FrameMetadata) -> None:
        """Non-blocking enqueue. Drops oldest on overflow."""
        try:
            self._queue.put_nowait(metadata)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest
                self._queue.put_nowait(metadata)
                self._dropped += 1
            except queue.Empty:
                pass

    @property
    def stats(self) -> dict[str, int]:
        return {
            "queued": self._queue.qsize(),
            "sent": self._sent,
            "dropped": self._dropped,
            "failed": self._failed,
        }

    def _run(self) -> None:
        settings = get_settings()
        url = settings.sentry_backend_url.rstrip("/") + "/api/v1/internal/live-metadata"
        headers = {
            "Authorization": f"Bearer {settings.sentry_backend_service_token}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=_REQUEST_TIMEOUT_SEC) as client:
            while not self._stop_event.is_set():
                batch = self._drain_batch()
                if not batch:
                    continue
                payload = {"frames": [m.model_dump(mode="json") for m in batch]}
                try:
                    r = client.post(url, json=payload, headers=headers)
                    if r.status_code >= 400:
                        self._failed += len(batch)
                        log.warning(
                            "emitter.http_error",
                            status=r.status_code,
                            body=r.text[:200],
                            batch=len(batch),
                        )
                    else:
                        self._sent += len(batch)
                except (httpx.RequestError, httpx.TimeoutException) as e:
                    self._failed += len(batch)
                    log.warning("emitter.request_failed", error=str(e), batch=len(batch))

    def _drain_batch(self) -> list[FrameMetadata]:
        """Collect up to _BATCH_SIZE items or wait up to _BATCH_TIMEOUT_SEC."""
        batch: list[FrameMetadata] = []
        deadline = time.monotonic() + _BATCH_TIMEOUT_SEC
        try:
            # Block briefly on first item to avoid busy-wait
            first = self._queue.get(timeout=_BATCH_TIMEOUT_SEC)
            batch.append(first)
        except queue.Empty:
            return batch

        while len(batch) < _BATCH_SIZE and time.monotonic() < deadline:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch
