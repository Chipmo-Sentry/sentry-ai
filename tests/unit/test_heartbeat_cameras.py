"""Heartbeat per-camera stream health (audit T12 #3) — pure derivation + HTTP read."""

from __future__ import annotations

from typing import Any

import httpx

from sentry_ai.heartbeat import _camera_health, _worker_stats


def _worker(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "camera_id": "cam-1",
        "rtsp_url": "rtsp://x",
        "running": True,
        "fps_capture": 5.0,
        "fps_inference": 4.8,
        "frames_total": 100,
        "detections_total": 10,
        "last_error": None,
    }
    base.update(overrides)
    return base


# --- _camera_health: status derivation ------------------------------------


def test_running_with_fps_is_ok() -> None:
    cams = _camera_health([_worker()])
    assert cams == [{"camera_id": "cam-1", "fps": 4.8, "status": "ok"}]


def test_running_zero_fps_no_error_is_stalled() -> None:
    cams = _camera_health([_worker(fps_inference=0.0)])
    assert cams[0]["status"] == "stalled"
    assert cams[0]["fps"] == 0.0


def test_read_failed_zero_fps_is_error() -> None:
    # camera_worker records "RTSP read failed" in last_error on read failure.
    cams = _camera_health([_worker(fps_inference=0.0, last_error="RTSP read failed")])
    assert cams[0]["status"] == "error"


def test_dead_worker_is_error_even_with_fps() -> None:
    cams = _camera_health([_worker(running=False, fps_inference=3.0)])
    assert cams[0]["status"] == "error"


def test_stale_error_with_live_fps_is_ok() -> None:
    # A recovered camera may still carry an old last_error — FPS wins.
    cams = _camera_health([_worker(fps_inference=2.5, last_error="old failure")])
    assert cams[0]["status"] == "ok"


def test_missing_fps_treated_as_zero() -> None:
    cams = _camera_health([_worker(fps_inference=None)])
    assert cams[0]["fps"] == 0.0
    assert cams[0]["status"] == "stalled"


def test_multiple_cameras_keep_order_and_ids() -> None:
    cams = _camera_health(
        [
            _worker(camera_id="cam-a"),
            _worker(camera_id="cam-b", fps_inference=0.0, last_error="RTSP read failed"),
        ]
    )
    assert [c["camera_id"] for c in cams] == ["cam-a", "cam-b"]
    assert [c["status"] for c in cams] == ["ok", "error"]


# --- _worker_stats: HTTP read keeps backward-compat sum + adds cameras -----


def _client_returning(payload: dict[str, Any] | None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if payload is None:
            raise httpx.ConnectError("app down")
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_worker_stats_sums_running_and_lists_all() -> None:
    payload = {
        "workers": [
            _worker(camera_id="cam-a", fps_inference=3.0),
            _worker(camera_id="cam-b", fps_inference=2.0),
            # Dead worker: excluded from sum/active but PRESENT in cameras.
            _worker(camera_id="cam-c", running=False, fps_inference=0.0),
        ]
    }
    with _client_returning(payload) as client:
        fps, active, cameras = _worker_stats(client)
    assert fps == 5.0
    assert active == 2
    assert cameras is not None
    assert [c["camera_id"] for c in cameras] == ["cam-a", "cam-b", "cam-c"]
    assert [c["status"] for c in cameras] == ["ok", "ok", "error"]


def test_worker_stats_empty_workers_is_empty_list_not_none() -> None:
    # Genuinely zero cameras (app reachable) → [], distinct from unknown (None).
    with _client_returning({"workers": []}) as client:
        fps, active, cameras = _worker_stats(client)
    assert (fps, active, cameras) == (0.0, 0, [])


def test_worker_stats_app_unreachable_reports_unknown() -> None:
    with _client_returning(None) as client:
        fps, active, cameras = _worker_stats(client)
    assert (fps, active) == (0.0, 0)
    assert cameras is None  # heartbeat omits the key → old behavior preserved
