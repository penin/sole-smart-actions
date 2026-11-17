"""Shared helpers: JSON I/O, hashing, seeding/determinism and environment capture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

TRACKED_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "openpyxl", "joblib", "PyYAML",
                    "torch", "keras", "transformers", "tokenizers", "safetensors", "huggingface-hub", "h5py",
                    "fastapi", "pytest"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(to_serializable(obj), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch; request deterministic kernels where supported."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def package_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


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
            return round(stat.ullTotalPhys / 1024**3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except Exception:
        return None


def _cpu_name() -> str | None:
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def environment_info() -> dict:
    """Software and hardware description recorded with every experiment."""
    env = {
        "os": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "total_memory_gb": _total_memory_gb(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "cuda_available": False, "cuda_version": None, "cudnn_version": None, "gpu": None,
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            env.update(cuda_available=True, cuda_version=torch.version.cuda,
                       cudnn_version=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(0),
                       gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    except ImportError:
        env["torch_version"] = "not installed"
    return env


def git_commit() -> str | None:
    try:
        from ..config import PROJECT_ROOT
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def command_line() -> str:
    """The command that produced an output, without personal absolute paths."""
    args = [Path(sys.argv[0]).name if os.path.isabs(sys.argv[0]) else sys.argv[0], *sys.argv[1:]]
    return "python " + " ".join(args)


def dir_size_bytes(paths) -> int:
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_file():
            total += p.stat().st_size
        elif p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total


def publish_dir(staging: Path, out: Path) -> Path:
    """Make ``out`` hold exactly the files in ``staging``, replacing files one by one.

    The folder itself is never deleted or renamed: on Windows, OneDrive, Explorer or an editor often keeps a handle
    on a folder, which makes ``rmtree`` + ``rename`` fail half-way. Individual file replacement is atomic per file."""
    staging, out = Path(staging), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keep = {p.relative_to(staging) for p in staging.rglob("*") if p.is_file()}
    for p in sorted(out.rglob("*"), reverse=True):
        if p.is_file() and p.relative_to(out) not in keep:
            p.unlink()
    for rel_path in sorted(keep):
        dst = out / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging / rel_path, dst)
    for p in sorted(out.rglob("*"), reverse=True):  # drop empty sub-folders left by stale files
        if p.is_dir() and not any(p.iterdir()):
            try:
                p.rmdir()
            except OSError:
                pass
    import shutil
    shutil.rmtree(staging, ignore_errors=True)
    return out
