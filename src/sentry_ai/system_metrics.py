"""Sample this AI server's resource usage (CPU / RAM / GPU) for telemetry.

Used by the heartbeat to report live load to the backend, which stores a
time-series the superadmin dashboard charts (live + 24h/7d/custom). Pure
data-out, never raises: on a CPU-only box (or if NVML is missing) the GPU
fields come back None and the rest still work.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

from sentry_ai.logging_setup import get_logger

log = get_logger("sentry_ai.system_metrics")

# psutil.cpu_percent needs a prior call to establish a baseline; we prime it
# once at import so the first real sample isn't a misleading 0.0.
try:
    import psutil

    psutil.cpu_percent(interval=None)
    _HAVE_PSUTIL = True
except Exception:  # noqa: BLE001 — optional dep / odd platform
    _HAVE_PSUTIL = False

# NVML (NVIDIA Management Library) — same numbers as nvidia-smi. Initialised
# lazily + once; absent on CPU-only/AMD boxes, which is fine.
_nvml_ready: bool | None = None


def _ensure_nvml() -> bool:
    global _nvml_ready
    if _nvml_ready is not None:
        return _nvml_ready
    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_ready = True
    except Exception as e:  # noqa: BLE001 — no NVIDIA GPU / driver
        log.debug("nvml.unavailable", error=str(e))
        _nvml_ready = False
    return _nvml_ready


@dataclass(slots=True)
class ResourceSample:
    """One instantaneous resource reading. GPU fields are None without a GPU."""

    cpu_pct: float | None = None
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None
    gpu_pct: int | None = None
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    gpu_temp_c: int | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def sample() -> ResourceSample:
    """Read CPU/RAM (psutil) + GPU (NVML) now. Never raises."""
    s = ResourceSample()
    if _HAVE_PSUTIL:
        try:
            s.cpu_pct = round(psutil.cpu_percent(interval=None), 1)
            vm = psutil.virtual_memory()
            s.ram_used_mb = int((vm.total - vm.available) / 1024 / 1024)
            s.ram_total_mb = int(vm.total / 1024 / 1024)
        except Exception as e:  # noqa: BLE001
            log.debug("psutil.sample_failed", error=str(e))

    if _ensure_nvml():
        try:
            import pynvml

            h = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0 (this box has one)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            s.gpu_pct = int(util.gpu)
            s.vram_used_mb = int(mem.used / 1024 / 1024)
            s.vram_total_mb = int(mem.total / 1024 / 1024)
            with contextlib.suppress(Exception):  # temp is optional
                s.gpu_temp_c = int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        except Exception as e:  # noqa: BLE001
            log.debug("nvml.sample_failed", error=str(e))

    return s
