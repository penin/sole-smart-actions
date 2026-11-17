"""Configuration loading. ``config/common.yaml`` holds the shared protocol; each model YAML is merged on top."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def resolve(path: str | Path) -> Path:
    """Resolve a repository-relative path (absolute paths are returned unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: str | Path) -> str:
    """Repository-relative POSIX path for reports and metadata (never a personal absolute path)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.name


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_common() -> dict:
    return _read(CONFIG_DIR / "common.yaml")


def load_config(model: str | None = None, *, smoke: bool = False) -> dict:
    """Common protocol merged with ``config/<model>.yaml``. With ``smoke`` the model's ``smoke`` block
    replaces the expensive settings (tiny grids) so the whole pipeline can be checked quickly."""
    cfg = load_common()
    if model:
        path = CONFIG_DIR / f"{model}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No configuration for model '{model}' ({rel(path)})")
        cfg = deep_merge(cfg, _read(path))
        if smoke and "smoke" in cfg:
            cfg = deep_merge(cfg, cfg["smoke"])
            # Smoke grids replace (not merge with) the full grids.
            for key in ("grid", "strategies", "C_grid"):
                if key in cfg["smoke"]:
                    cfg[key] = copy.deepcopy(cfg["smoke"][key])
    cfg["smoke"] = bool(smoke)
    return cfg


def config_files(model: str) -> list[Path]:
    return [CONFIG_DIR / "common.yaml", CONFIG_DIR / f"{model}.yaml"]


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
