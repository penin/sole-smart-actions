"""Train one model under the shared protocol and write a self-describing artifact directory.

Primary protocol: fit on the 2,160 training commands, select on the 270 validation commands.
Secondary protocols (clearly labelled, never part of the primary comparison):
  * ``trainval_refit`` - the teammates' original recipe: refit the selected configuration on train+validation
  * ablations (Custom Transformer): ``no_attention`` / ``no_glove`` with the primary configuration
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

from .config import config_files, load_config, rel, resolve
from .data import load_splits
from .models import create_model, model_class
from .utils import (command_line, environment_info, git_commit, load_json, publish_dir, save_json, set_seed,
                    sha256_file, utc_now)

CARD = "model_card.json"
PROTOCOLS = ("primary", "trainval_refit")


def artifact_dir(name: str, *, protocol: str = "primary", variant: str | None = None,
                 root: str | Path | None = None) -> Path:
    base = resolve(root or load_config()["paths"]["artifacts_dir"])
    if protocol == "primary" and not variant:
        return base / name
    suffix = variant or protocol
    return base / "secondary" / f"{name}_{suffix}"


def run_label(protocol: str, variant: str | None) -> str:
    return variant or protocol


def _file_records(directory: Path) -> list[dict]:
    return [{"path": p.relative_to(directory).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != CARD]


def train_model(name: str, *, smoke: bool = False, protocol: str = "primary", variant: str | None = None,
                artifacts_root: str | Path | None = None) -> Path:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    cfg = load_config(name, smoke=smoke)
    splits, fingerprint = load_splits(cfg)            # validates the dataset; raises on any failure
    set_seed(int(cfg["model"]["seed"]))
    out = artifact_dir(name, protocol=protocol, variant=variant, root=artifacts_root)
    label = run_label(protocol, variant)
    print(f"[train] {name} ({label}{', SMOKE' if smoke else ''}) -> {rel(out)}", flush=True)

    model = create_model(name, cfg)
    started = time.perf_counter()
    if variant:
        ablations = cfg.get("ablations") or {}
        if variant not in ablations:
            raise ValueError(f"{name} has no ablation '{variant}' (available: {sorted(ablations)})")
        primary = load_json(artifact_dir(name, root=artifacts_root) / CARD)
        params = {k: v for k, v in primary["selection"]["params"].items() if k != "epochs"}
        model.fit(splits.train, splits.validation, fixed_config=params, override=ablations[variant])
    else:
        model.fit(splits.train, splits.validation)
    if protocol == "trainval_refit":
        model.refit(pd.concat([splits.train, splits.validation], ignore_index=True))
    wall = time.perf_counter() - started

    staging = out.with_name(out.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    model.save(staging)
    if model.search_table is not None:
        model.search_table.to_csv(staging / "hyperparameter_search.csv", index=False)
    card = {
        "model": name, "display_name": cfg["model"]["display_name"], "original_author": cfg["model"]["original_author"],
        "protocol": protocol, "variant": variant, "run_label": label, "smoke_test": smoke,
        "primary_comparison": protocol == "primary" and not variant and not smoke,
        "trained_on": model.trained_on, "selected_on": ["validation"],
        "created_utc": utc_now(), "command": command_line(), "git_commit": git_commit(),
        "dataset": fingerprint, "seed": cfg["model"]["seed"],
        "config_files": {rel(p): sha256_file(p) for p in config_files(name)}, "config": cfg,
        "selection": model.selection, "metadata": model.get_model_metadata(),
        "training_seconds_final_fit": model.training_seconds, "training_wall_seconds_incl_search": wall,
        "environment": environment_info(),
    }
    card["files"] = _file_records(staging)
    card["size_bytes"] = sum(f["bytes"] for f in card["files"])
    save_json(card, staging / CARD)
    publish_dir(staging, out)
    print(f"[train] {name}: selected {model.selection.get('params')} in {wall:.1f}s", flush=True)
    return out


def load_trained(name: str, directory: str | Path | None = None, *, cfg: dict | None = None):
    """Load a trained artifact through the common interface, verifying file hashes against its model card."""
    directory = Path(directory) if directory else artifact_dir(name)
    card_path = directory / CARD
    if not card_path.exists():
        raise FileNotFoundError(f"No trained {name} model in {rel(directory)} (missing {CARD}); "
                                f"run: python -m scripts.train --model {name}")
    card = load_json(card_path)
    for f in card["files"]:
        p = directory / f["path"]
        if not p.exists():
            raise FileNotFoundError(f"{rel(p)} listed in the model card is missing (Git LFS not pulled?)")
        if p.stat().st_size < 1024 and f["bytes"] > 1024 and p.read_bytes().startswith(b"version https://git-lfs"):
            raise FileNotFoundError(f"{rel(p)} is a Git LFS pointer; run: git lfs pull")
    cfg = cfg or load_config(name, smoke=card.get("smoke_test", False))
    return model_class(name).load(directory, cfg), card


def verify_artifact_hashes(directory: str | Path) -> dict:
    directory = Path(directory)
    card = load_json(directory / CARD)
    bad = [f["path"] for f in card["files"] if sha256_file(directory / f["path"]) != f["sha256"]]
    return {"directory": rel(directory), "files": len(card["files"]), "mismatched": bad, "ok": not bad}
