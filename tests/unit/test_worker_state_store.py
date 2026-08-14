"""Worker-state persistence (live_worker/state_store.py): a node restart must
replay exactly the workers that were running — started ones survive, stopped
ones don't, and a corrupt file degrades to "nothing to replay", never a crash."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentry_ai.live_worker import state_store
from sentry_ai.settings import get_settings


@pytest.fixture()
def state_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "worker_state.json"
    monkeypatch.setenv("WORKER_STATE_PATH", str(p))
    get_settings.cache_clear()
    yield p
    get_settings.cache_clear()


def _spec(rtsp: str) -> dict:
    return {
        "rtsp_url": rtsp,
        "frame_skip": 3,
        "store_id": "store-1",
        "alert_threshold_pct": 40.0,
        "zones": [{"name": "касс", "points": [[0, 0], [1, 0], [1, 1]]}],
    }


def test_started_specs_survive_and_include_camera_id(state_path: Path) -> None:
    state_store.record_started("cam_a", _spec("rtsp://127.0.0.1:8554/cam_a"))
    state_store.record_started("cam_b", _spec("rtsp://127.0.0.1:8554/cam_b"))

    specs = {s["camera_id"]: s for s in state_store.load_specs()}
    assert set(specs) == {"cam_a", "cam_b"}
    # The saved dict is the full start_camera kwargs — zones included, so a
    # restored worker re-arms its behavior engine identically.
    assert specs["cam_a"]["zones"][0]["name"] == "касс"
    assert specs["cam_b"]["rtsp_url"].endswith("/cam_b")


def test_stopped_worker_is_forgotten(state_path: Path) -> None:
    state_store.record_started("cam_a", _spec("rtsp://x/a"))
    state_store.record_started("cam_b", _spec("rtsp://x/b"))
    state_store.record_stopped("cam_a")

    assert [s["camera_id"] for s in state_store.load_specs()] == ["cam_b"]
    # Stopping something unknown is a no-op, not an error.
    state_store.record_stopped("never-existed")
    assert [s["camera_id"] for s in state_store.load_specs()] == ["cam_b"]


def test_restart_overwrites_same_camera(state_path: Path) -> None:
    state_store.record_started("cam_a", _spec("rtsp://old/a"))
    state_store.record_started("cam_a", _spec("rtsp://new/a"))

    specs = state_store.load_specs()
    assert len(specs) == 1
    assert specs[0]["rtsp_url"] == "rtsp://new/a"


def test_corrupt_or_missing_file_degrades_to_empty(state_path: Path) -> None:
    assert state_store.load_specs() == []  # missing file
    state_path.write_text("{not json", encoding="utf-8")
    assert state_store.load_specs() == []  # corrupt file
    # And writing after corruption recovers cleanly.
    state_store.record_started("cam_a", _spec("rtsp://x/a"))
    assert [s["camera_id"] for s in state_store.load_specs()] == ["cam_a"]
