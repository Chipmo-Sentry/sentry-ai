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
    """One instantaneous resource reading.

    The bare fields are WHOLE-MACHINE (psutil/NVML report the entire box, every
    process + OS). The ``sentry_*`` fields isolate just THIS project's footprint
    (sentry-ai + ingest/mediamtx + ollama + cloudflared and their children), so
    the dashboard can show "my project's usage" vs "the whole server". GPU fields
    are None without a GPU; ``sentry_gpu_pct`` is intentionally absent — NVML only
    exposes GPU *utilisation* for the whole device, not per process (only VRAM is
    per-process).
    """

    cpu_pct: float | None = None
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None
    gpu_pct: int | None = None
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    gpu_temp_c: int | None = None
    # Sentry-project-only (process-scoped) usage:
    sentry_cpu_pct: float | None = None
    sentry_ram_mb: int | None = None
    sentry_vram_mb: int | None = None
    # Per-component CPU/RAM breakdown (sentry-ai / Ollama / MediaMTX / Tunnel) so
    # the dashboard can show WHO uses what. Each: {name, cpu_pct, ram_mb}. None on
    # a box without psutil. Not part of the numeric metrics time-series.
    components: list[dict[str, float | int | str | None]] | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# Persistent psutil.Process cache so per-process cpu_percent() measures load over
# the interval between heartbeats (a fresh Process always reports 0.0 first).
_proc_cache: dict[int, object] = {}


def _category(name: str, cmd: str) -> str | None:
    """Group a process into a dashboard-facing component, or None if not ours.
    Substring matches so Ollama's model runner (ollama_llama_server) counts too."""
    if "mediamtx" in name:
        return "MediaMTX"
    if "ollama" in name:
        return "Ollama"
    if "cloudflared" in name:
        return "Tunnel"
    if name.startswith(("python", "pythonw", "uv")) and "sentry_ai" in cmd:
        return "sentry-ai"
    return None


def _proc_category(p: object) -> str | None:
    try:
        name = (p.info.get("name") or "").lower()  # type: ignore[attr-defined]
        cmd = " ".join(p.info.get("cmdline") or []).lower()  # type: ignore[attr-defined]
        return _category(name, cmd)
    except Exception:  # noqa: BLE001 — process vanished / access denied
        return None


def _sample_sentry(s: ResourceSample) -> None:
    """Fill the ``sentry_*`` totals AND the per-component breakdown (s.components)
    with this project's process-scoped usage, in a single psutil pass."""
    if not _HAVE_PSUTIL:
        return
    global _proc_cache
    pids: set[int] = set()
    try:
        matched: dict[int, object] = {}
        cats: dict[int, str] = {}
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            cat = _proc_category(p)
            if cat is not None:
                proc = _proc_cache.get(p.pid) or p
                matched[p.pid] = proc
                cats[p.pid] = cat
                pids.add(p.pid)
                # include children (e.g. ollama's GPU runner) for VRAM matching
                with contextlib.suppress(Exception):
                    for c in p.children(recursive=True):
                        pids.add(c.pid)
        _proc_cache = matched  # drop dead pids

        ncpu = psutil.cpu_count() or 1
        # Accumulate per component: {category: [cpu_sum, ram_bytes]}.
        groups: dict[str, list[float]] = {}
        for pid, proc in matched.items():
            c = 0.0
            r = 0
            with contextlib.suppress(Exception):
                c = proc.cpu_percent(None)  # type: ignore[attr-defined]
            with contextlib.suppress(Exception):
                r = int(proc.memory_info().rss)  # type: ignore[attr-defined]
            g = groups.setdefault(cats[pid], [0.0, 0.0])
            g[0] += c
            g[1] += r
        s.sentry_cpu_pct = round(sum(g[0] for g in groups.values()) / ncpu, 1)
        s.sentry_ram_mb = int(sum(g[1] for g in groups.values()) / 1024 / 1024)
        s.components = [
            {
                "name": name,
                "cpu_pct": round(g[0] / ncpu, 1),
                "ram_mb": int(g[1] / 1024 / 1024),
            }
            for name, g in sorted(groups.items())
        ]
    except Exception as e:  # noqa: BLE001
        log.debug("psutil.sentry_sample_failed", error=str(e))

    # Per-process VRAM via NVML (utilisation can't be isolated per-process).
    if _ensure_nvml() and pids:
        try:
            import pynvml

            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            procs = []
            with contextlib.suppress(Exception):
                procs += list(pynvml.nvmlDeviceGetComputeRunningProcesses(h))
            with contextlib.suppress(Exception):
                procs += list(pynvml.nvmlDeviceGetGraphicsRunningProcesses(h))
            vram = 0
            for gp in procs:
                used = getattr(gp, "usedGpuMemory", None)
                if used and gp.pid in pids:
                    vram += int(used)
            s.sentry_vram_mb = int(vram / 1024 / 1024)
        except Exception as e:  # noqa: BLE001
            log.debug("nvml.sentry_sample_failed", error=str(e))


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

    _sample_sentry(s)
    return s
