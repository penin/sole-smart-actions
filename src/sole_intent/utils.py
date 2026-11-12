"""Shared helpers: project paths, configuration, hashing, JSON I/O and environment capture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "baseline.yaml"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRACKED_PACKAGES = [
    "scikit-learn", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "openpyxl", "joblib", "PyYAML", "pytest",
]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> dict:
    with open(resolve_path(path or DEFAULT_CONFIG), encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs() -> None:
    for d in (METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, REPORTS_DIR, MODELS_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OutputPaths:
    """Where one experiment writes. Defaults reproduce the original Baseline V1 (V4 data) locations."""
    results: Path
    models: Path
    processed: Path

    @property
    def metrics(self) -> Path:
        return self.results / "metrics"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def predictions(self) -> Path:
        return self.results / "predictions"

    @property
    def reports(self) -> Path:
        return self.results / "reports"

    def ensure(self) -> None:
        for d in (self.metrics, self.figures, self.predictions, self.reports, self.models, self.processed):
            d.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path) -> str:
        """Project-relative POSIX path for reports (absolute if the path is outside the project)."""
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return str(resolved).replace("\\", "/")


def output_paths(cfg: dict) -> OutputPaths:
    out = cfg.get("outputs") or {}
    return OutputPaths(results=resolve_path(out.get("results_dir", "results")),
                       models=resolve_path(out.get("models_dir", "models")),
                       processed=resolve_path(out.get("processed_dir", "data/processed")))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_serializable(obj: Any) -> Any:
    """Convert numpy / pathlib objects to JSON types. NaN and inf become None (JSON null),
    so undefined metrics are explicit rather than silently replaced by zero."""
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(obj), f, indent=2, allow_nan=False)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _total_memory_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / 1024**3, 2)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
    except Exception:
        return None


def environment_info() -> dict:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "total_memory_gb": _total_memory_gb(),
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version.split()[0],
        "hardware_accelerator": "none (CPU only)",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
