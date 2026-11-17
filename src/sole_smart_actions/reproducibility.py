"""Environment verification, result-schema checks and reproduction comparisons.

Classical models (BoW, TF-IDF) must reproduce exactly (identical predictions; probabilities within 1e-9).
Neural models are compared with tolerances: deterministic kernels are requested, but bit-identical weights
across GPUs, drivers or library builds are not guaranteed.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, rel, resolve
from .utils import environment_info, load_json

EXACT = {"bow", "tfidf"}
NEURAL_TOLERANCE = {"raw_test_accuracy_abs": 3 / 270, "test_macro_f1_abs": 0.012,
                    "test_prediction_agreement_min": 0.97, "default_threshold_abs": 0.05}
REQUIRED_METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_roc_auc_ovr",
                        "per_class", "confusion_matrix", "class_order", "n_samples"]
REQUIRED_SWEEP_COLUMNS = ["threshold_label", "threshold", "known_coverage", "accepted_known", "accepted_known_correct",
                          "accuracy_among_accepted", "overall_correct_routing_rate", "rejected_known",
                          "unknown_rejection_rate", "unknown_as_help_support", "unknown_safe_fallback_rate",
                          "unknown_to_financial"]


def verify_environment(lockfile: str | Path = "requirements.lock") -> dict:
    """Compare installed packages with the lockfile pins; report Python/CUDA."""
    path = resolve(lockfile)
    pins, mismatches = {}, []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
            if m:
                pins[m.group(1).lower()] = m.group(2)
    for name, want in pins.items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            mismatches.append({"package": name, "pinned": want, "installed": None})
            continue
        if got.split("+")[0] != want.split("+")[0]:
            mismatches.append({"package": name, "pinned": want, "installed": got})
    env = environment_info()
    want_py = (path.read_text(encoding="utf-8").split("python_version:")[1].split()[0]
               if path.exists() and "python_version:" in path.read_text(encoding="utf-8") else None)
    py_ok = want_py is None or sys.version.startswith(want_py.rsplit(".", 1)[0])
    return {"lockfile": rel(path), "pinned_packages": len(pins), "mismatches": mismatches,
            "python": env["python"], "python_expected": want_py, "python_ok": py_ok,
            "cuda_available": env["cuda_available"], "gpu": env["gpu"],
            "ok": not mismatches and py_ok, "environment": env}


def verify_result_schema(results_dir: str | Path, model: str) -> dict:
    from .evaluation.runner import STANDARD_FILES
    d = Path(results_dir) / model
    problems = [f"missing {f}" for f in STANDARD_FILES if not (d / f).exists()]
    if not problems:
        labels = load_config()["labels"]
        for split in ("train", "validation", "test"):
            m = load_json(d / f"metrics_{split}.json")
            problems += [f"metrics_{split}.json lacks {k}" for k in REQUIRED_METRIC_KEYS if k not in m]
            if m.get("class_order") != labels:
                problems.append(f"metrics_{split}.json class order differs from canonical order")
        sweep = pd.read_csv(d / "threshold_sweep.csv")
        problems += [f"threshold_sweep.csv lacks {c}" for c in REQUIRED_SWEEP_COLUMNS if c not in sweep]
        want = ["none", "default", "0.25", "0.30", "0.35", "0.40", "0.50", "0.60"]
        if sweep["threshold_label"].astype(str).tolist() != want:
            problems.append(f"threshold_sweep.csv rows {sweep['threshold_label'].tolist()} != {want}")
        preds = pd.read_csv(d / "test_predictions.csv")
        if [c for c in preds if c.startswith("prob_")] != [f"prob_{c}" for c in labels]:
            problems.append("test_predictions.csv probability columns not in canonical order")
        probs = preds[[f"prob_{c}" for c in labels]].to_numpy()
        if not np.allclose(probs.sum(1), 1, atol=1e-5):
            problems.append("test probabilities do not sum to 1")
        meta = load_json(d / "experiment_metadata.json")
        if abs(meta["headline"]["raw_test_accuracy"] - preds["correct_raw"].mean()) > 1e-12:
            problems.append("headline accuracy differs from test_predictions.csv")
    return {"model": model, "directory": rel(d), "ok": not problems, "problems": problems}


def compare_results(reference: str | Path, reproduced: str | Path, model: str) -> dict:
    """Compare a reproduced evaluation against the published one."""
    a, b = Path(reference) / model, Path(reproduced) / model
    ma, mb = load_json(a / "experiment_metadata.json"), load_json(b / "experiment_metadata.json")
    pa, pb = pd.read_csv(a / "test_predictions.csv"), pd.read_csv(b / "test_predictions.csv")
    labels = [c for c in pa if c.startswith("prob_")]
    out = {"model": model, "mode": "exact" if model in EXACT else "tolerance",
           "reference": rel(a), "reproduced": rel(b),
           "raw_test_accuracy": [ma["headline"]["raw_test_accuracy"], mb["headline"]["raw_test_accuracy"]],
           "test_macro_f1": [ma["headline"]["test_macro_f1"], mb["headline"]["test_macro_f1"]],
           "default_threshold": [ma["headline"]["default_threshold"], mb["headline"]["default_threshold"]],
           "hyperparameters": [ma["hyperparameters"], mb["hyperparameters"]],
           "test_prediction_agreement": float((pa["predicted_intent"] == pb["predicted_intent"]).mean()),
           "max_abs_probability_difference": float(np.abs(pa[labels].to_numpy() - pb[labels].to_numpy()).max())}
    if out["mode"] == "exact":
        out["ok"] = (out["test_prediction_agreement"] == 1.0 and out["max_abs_probability_difference"] <= 1e-9
                     and out["hyperparameters"][0] == out["hyperparameters"][1])
        out["criteria"] = "identical predictions, probabilities within 1e-9, same selected hyperparameters"
    else:
        t = NEURAL_TOLERANCE
        out["ok"] = (abs(out["raw_test_accuracy"][0] - out["raw_test_accuracy"][1]) <= t["raw_test_accuracy_abs"] + 1e-12
                     and abs(out["test_macro_f1"][0] - out["test_macro_f1"][1]) <= t["test_macro_f1_abs"]
                     and out["test_prediction_agreement"] >= t["test_prediction_agreement_min"]
                     and abs(out["default_threshold"][0] - out["default_threshold"][1]) <= t["default_threshold_abs"])
        out["criteria"] = (f"raw test accuracy within {t['raw_test_accuracy_abs']:.4f} (3 commands), macro F1 within "
                           f"{t['test_macro_f1_abs']}, >= {t['test_prediction_agreement_min']:.0%} identical test "
                           f"predictions, default threshold within {t['default_threshold_abs']}")
        out["bit_identical"] = out["max_abs_probability_difference"] == 0.0
    return out


def write_reproducibility_report(path: Path, *, env: dict, schema: list[dict], comparisons: list[dict] | None,
                                 hashes: list[dict], mode: str, data_report: dict, notes: list[str] = ()) -> Path:
    L = ["# Sole Smart Actions - reproducibility report", "",
         f"Mode: **{mode}**. Environment: Python {env['python']}, "
         f"{env['environment']['os']}, CPU {env['environment']['cpu']}, GPU {env['gpu'] or 'none'} "
         f"(CUDA {env['environment']['cuda_version'] or 'n/a'}), torch {env['environment'].get('torch_version')}, "
         f"scikit-learn {env['environment']['packages'].get('scikit-learn')}.", "",
         "## Checks", "", "| Check | Result | Detail |", "|---|---|---|",
         f"| Dataset validation ({data_report['n_checks']} checks) | {'PASS' if data_report['passed'] else 'FAIL'} | "
         f"SHA-256 `{data_report['dataset']['sha256'][:16]}...` |",
         f"| Environment matches `{env['lockfile']}` | {'PASS' if env['ok'] else 'WARN'} | "
         f"{len(env['mismatches'])} package mismatches; Python {env['python']} (expected {env['python_expected']}) |"]
    for h in hashes:
        L.append(f"| Artifact hashes `{h['directory']}` | {'PASS' if h['ok'] else 'FAIL'} | {h['files']} files; "
                 f"mismatched: {h['mismatched'] or 'none'} |")
    for s in schema:
        L.append(f"| Result schema `{s['directory']}` | {'PASS' if s['ok'] else 'FAIL'} | "
                 f"{'; '.join(s['problems']) or 'all standard files and fields present'} |")
    if comparisons:
        L += ["", "## Reproduced vs published", "",
              "| Model | Mode | Raw test accuracy (published -> reproduced) | Macro F1 | Same test predictions | "
              "Max prob. difference | Result |", "|---|---|---|---|---|---|---|"]
        for c in comparisons:
            L.append(f"| {c['model']} | {c['mode']} | {c['raw_test_accuracy'][0]:.4f} -> {c['raw_test_accuracy'][1]:.4f} | "
                     f"{c['test_macro_f1'][0]:.4f} -> {c['test_macro_f1'][1]:.4f} | {c['test_prediction_agreement']:.1%} | "
                     f"{c['max_abs_probability_difference']:.2e} | {'PASS' if c['ok'] else 'FAIL'} |")
        L += ["", "Criteria: " + "; ".join(sorted({f"{c['mode']}: {c['criteria']}" for c in comparisons}))]
    if notes:
        L += ["", "## Notes", "", *[f"- {n}" for n in notes]]
    L += ["", "Neural networks: deterministic PyTorch kernels are requested (`torch.use_deterministic_algorithms`, "
          "`CUBLAS_WORKSPACE_CONFIG`, fixed seeds), but bit-identical weights across GPUs, drivers, CUDA/cuDNN or "
          "library versions are not guaranteed, so they are verified with tolerances.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")
    return path
