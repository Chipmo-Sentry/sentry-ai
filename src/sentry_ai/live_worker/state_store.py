"""Persist live-worker specs across process restarts.

The backend provisions workers only on camera CRUD / agent re-register, so a
node restart (vast host maintenance, supervisor restart, crash) silently
dropped every worker until a human touched a camera or the agent re-registered
— the store looked "streaming" while AI was dark (bit us 2026-08-11). The
manager records every started worker here and removes intentionally stopped
ones; main.py replays the file at startup.

Best-effort by design: a corrupt/unwritable state file must never break
start/stop of the live pipeline, so every public function swallows I/O errors
after logging them.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from sentry_ai.logging_setup import get_logger
from sentry_ai.settings import get_settings

log = get_logger("sentry_ai.live_worker.state_store")

_lock = threading.Lock()


def _path() -> str:
    return get_settings().worker_state_path


def _read() -> dict[str, dict[str, Any]]:
    try:
        with open(_path(), encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        log.warning("state_store.read_failed", path=_path(), exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _write(data: dict[str, dict[str, Any]]) -> None:
    # Atomic replace so a crash mid-write can't leave a truncated file.
    path = _path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def record_started(camera_id: str, spec: dict[str, Any]) -> None:
    """Remember a started worker's full start_camera kwargs."""
    with _lock:
        try:
            data = _read()
            data[camera_id] = spec
            _write(data)
        except OSError:
            log.warning("state_store.write_failed", camera_id=camera_id, exc_info=True)


def record_stopped(camera_id: str) -> None:
    """Forget an INTENTIONALLY stopped worker (deprovision). Shutdown must NOT
    call this — the whole point is surviving restarts."""
    with _lock:
        try:
            data = _read()
            if data.pop(camera_id, None) is not None:
                _write(data)
        except OSError:
            log.warning("state_store.write_failed", camera_id=camera_id, exc_info=True)


def load_specs() -> list[dict[str, Any]]:
    """Saved start_camera kwarg dicts (camera_id included), for startup replay."""
    with _lock:
        return [{"camera_id": cid, **spec} for cid, spec in _read().items()]
