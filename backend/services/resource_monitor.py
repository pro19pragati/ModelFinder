"""Resource inspection: CPU/RAM availability, hard-limit checks, measurement.

Works with or without ``psutil`` (falls back to stdlib / ctypes) so the backend
runs on a plain Python install. Every value produced here is clearly marked as
either *measured* (during a training run) or *estimated* (before a run).
"""

from __future__ import annotations

import ctypes
import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:  # pragma: no cover - Unix only module
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

_IS_WINDOWS = platform.system().lower().startswith("win")


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def available_cpu_count() -> int:
    """Logical CPUs the agent may use."""
    if psutil is not None:
        try:
            return max(1, psutil.cpu_count(logical=True) or 1)
        except Exception:  # pragma: no cover
            pass
    return max(1, os.cpu_count() or 1)


def _windows_memory() -> Optional[Dict[str, float]]:  # pragma: no cover
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        total_mb = status.ullTotalPhys / (1024 ** 2)
        avail_mb = status.ullAvailPhys / (1024 ** 2)
        return {
            "total_mb": round(total_mb, 1),
            "available_mb": round(avail_mb, 1),
            "used_percent": round((1 - avail_mb / total_mb) * 100, 1) if total_mb else 0.0,
        }
    except Exception:
        return None


def _posix_memory() -> Optional[Dict[str, float]]:  # pragma: no cover
    try:
        total_mb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
        avail_mb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
        return {
            "total_mb": round(total_mb, 1),
            "available_mb": round(avail_mb, 1),
            "used_percent": round((1 - avail_mb / total_mb) * 100, 1) if total_mb else 0.0,
        }
    except Exception:
        return None


def memory_info() -> Dict[str, float]:
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return {
                "total_mb": round(vm.total / (1024 ** 2), 1),
                "available_mb": round(vm.available / (1024 ** 2), 1),
                "used_percent": float(vm.percent),
            }
        except Exception:  # pragma: no cover
            pass
    info = _windows_memory() if _IS_WINDOWS else _posix_memory()
    if info:
        return info
    return {"total_mb": 0.0, "available_mb": 0.0, "used_percent": 0.0}


def process_memory_mb() -> float:
    """Current resident memory of this process in MB (0 when unavailable)."""
    if psutil is not None:
        try:
            return round(psutil.Process().memory_info().rss / (1024 ** 2), 1)
        except Exception:  # pragma: no cover
            pass
    if _IS_WINDOWS:  # pragma: no cover
        try:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return round(counters.WorkingSetSize / (1024 ** 2), 1)
        except Exception:
            return 0.0
    if resource is not None:  # pragma: no cover
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            factor = 1024 if platform.system().lower() != "darwin" else 1024 * 1024
            return round(usage / factor, 1)
        except Exception:
            return 0.0
    return 0.0


def resource_snapshot() -> Dict[str, Any]:
    """Current machine resources, used for budgeting and shown in the UI."""
    mem = memory_info()
    cpu_count = available_cpu_count()
    try:
        load_avg = round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None
    except Exception:  # pragma: no cover
        load_avg = None
    return {
        "cpu_count": cpu_count,
        "memory_total_mb": mem["total_mb"],
        "memory_available_mb": mem["available_mb"],
        "memory_used_percent": mem["used_percent"],
        "process_memory_mb": process_memory_mb(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "load_average_1m": load_avg,
        "measured": True,
        "source": "psutil" if psutil is not None else "stdlib",
    }


def estimate_training_memory_mb(n_rows: int, n_columns: int, model_complexity: float) -> float:
    """Rough, clearly-labelled memory estimate for a single training run."""
    feature_matrix_mb = (n_rows * max(1, n_columns) * 8) / (1024 ** 2)
    return round(feature_matrix_mb * (3.0 + 0.5 * model_complexity), 1)


def check_hard_limits(
    n_rows: int,
    n_columns: int,
    file_size_mb: float,
    model_complexity: float = 1.0,
    required_memory_mb: Optional[float] = None,
) -> List[Dict[str, str]]:
    """Return hard-limit violations (empty list means the dataset is acceptable)."""
    violations: List[Dict[str, str]] = []
    if file_size_mb > settings.max_upload_size_mb:
        violations.append(
            {
                "code": "file_too_large",
                "message": (
                    f"File size {file_size_mb:.1f} MB exceeds the configured limit of "
                    f"{settings.max_upload_size_mb:.0f} MB."
                ),
            }
        )
    if n_rows > settings.max_rows:
        violations.append(
            {
                "code": "too_many_rows",
                "message": (
                    f"{n_rows:,} rows exceed the configured hard limit of "
                    f"{settings.max_rows:,} rows."
                ),
            }
        )
    if n_columns > settings.max_columns:
        violations.append(
            {
                "code": "too_many_columns",
                "message": (
                    f"{n_columns:,} columns exceed the configured hard limit of "
                    f"{settings.max_columns:,} columns."
                ),
            }
        )
    mem = memory_info()
    if mem["available_mb"] > 0:
        need = required_memory_mb or estimate_training_memory_mb(
            n_rows, n_columns, model_complexity
        )
        budget_mb = mem["available_mb"] * 0.7
        if need > budget_mb:
            violations.append(
                {
                    "code": "insufficient_memory",
                    "message": (
                        f"Estimated training memory {need:.0f} MB exceeds 70% of available RAM "
                        f"({budget_mb:.0f} MB of {mem['available_mb']:.0f} MB free)."
                    ),
                }
            )
    return violations


@dataclass
class ResourceUsage:
    """Measured resource consumption of one training / evaluation run."""

    wall_seconds: float
    cpu_seconds: float
    cpu_utilization_pct: float
    peak_memory_mb: float
    rss_delta_mb: float
    measured: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "wall_seconds": round(self.wall_seconds, 4),
            "cpu_seconds": round(self.cpu_seconds, 4),
            "cpu_utilization_pct": round(self.cpu_utilization_pct, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "rss_delta_mb": round(self.rss_delta_mb, 1),
            "measured": self.measured,
        }


class ResourceTracker:
    """Measures wall time, CPU time and peak memory for a block of work."""

    def __init__(self, label: str = "training") -> None:
        self.label = label
        self.usage = ResourceUsage(0.0, 0.0, 0.0, 0.0, 0.0)
        self._wall_start = 0.0
        self._cpu_start = 0.0
        self._rss_start = 0.0
        self._peak_rss = 0.0

    def __enter__(self) -> "ResourceTracker":
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        self._rss_start = process_memory_mb()
        self._peak_rss = self._rss_start
        return self

    def sample(self) -> None:
        """Optional manual memory sample (used between pipeline stages)."""
        self._peak_rss = max(self._peak_rss, process_memory_mb())

    def __exit__(self, exc_type, exc, tb) -> bool:
        wall = time.perf_counter() - self._wall_start
        cpu = time.process_time() - self._cpu_start
        self._peak_rss = max(self._peak_rss, process_memory_mb())
        cpus = available_cpu_count()
        self.usage = ResourceUsage(
            wall_seconds=wall,
            cpu_seconds=cpu,
            cpu_utilization_pct=(cpu / wall / cpus * 100) if wall > 0 else 0.0,
            peak_memory_mb=self._peak_rss,
            rss_delta_mb=max(0.0, self._peak_rss - self._rss_start),
        )
        return False

