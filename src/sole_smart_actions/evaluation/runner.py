"""Evaluate one trained model with the shared protocol and write the standard result files.

Every number is computed here from the model's actual predictions on the fixed V5 splits:
    metrics_train.json, metrics_validation.json, metrics_test.json, classification_report.csv,
    confusion_matrix.csv/.png, roc_auc.json, roc_curves.png, test_predictions.csv, unknown_predictions.csv,
    threshold_sweep.csv, latency_metrics.json, experiment_metadata.json, evaluation_report.md
plus secondary analyses (threshold_sweep_dense.csv, secondary_analyses.json, robustness_typos.json).
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from sole_intent.evaluate import evaluate_predictions, measure_latency

from ..config import load_config, rel, resolve
from ..data import load_splits
from ..training import CARD, artifact_dir, load_trained
from ..utils import command_line, environment_info, git_commit, publish_dir, save_json, sha256_file, utc_now
from . import plots
from .report import write_report
from .thresholds import (calibrate_threshold, dense_sweep, temperature_calibration, threshold_sweep,
                         three_tier_routing, threshold_for_coverage)

SCHEMA_VERSION = "1.0"
STANDARD_FILES = ["metrics_train.json", "metrics_validation.json", "metrics_test.json", "classification_report.csv",
                  "confusion_matrix.csv", "confusion_matrix.png", "roc_auc.json", "roc_curves.png",
                  "test_predictions.csv", "unknown_predictions.csv", "threshold_sweep.csv", "latency_metrics.json",
                  "experiment_metadata.json", "evaluation_report.md"]


def results_dir(name: str, label: str = "primary", root: str | Path | None = None) -> Path:
    base = resolve(root or load_config()["paths"]["results_dir"])
    return base / name if label == "primary" else base / "secondary" / f"{name}_{label}"


def add_typos(text: str, rng: random.Random) -> str:
    """The TF-IDF teammate's ``add_typos``: one keyboard slip in one word longer than 3 characters."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) > 3]
    if not idx:
        return text
    i = rng.choice(idx)
    w = list(words[i])
    p = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "dup", "sub"])
    if op == "swap":
        w[p], w[p + 1] = w[p + 1], w[p]
    elif op == "drop":
        del w[p]
    elif op == "dup":
        w.insert(p, w[p])
    else:
        w[p] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    words[i] = "".join(w)
    return " ".join(words)


def _prediction_frame(df: pd.DataFrame, proba: np.ndarray, classes: list[str], threshold: float) -> pd.DataFrame:
    order = np.argsort(-proba, axis=1)
    pred = np.asarray(classes, dtype=object)[order[:, 0]]
    conf = proba.max(1)
    out = df[["row_id", "excel_row", "text", "intent", "split"]].rename(columns={"intent": "true_intent"}).copy()
    out["predicted_intent"] = pred
    out["confidence"] = conf
    out["second_intent"] = np.asarray(classes, dtype=object)[order[:, 1]]
    out["second_confidence"] = proba[np.arange(len(proba)), order[:, 1]]
    out["accepted_at_default_threshold"] = conf >= threshold
    out["final_decision_at_default_threshold"] = np.where(conf >= threshold, pred, "UNKNOWN_INTENT")
    out["correct_raw"] = out["predicted_intent"] == out["true_intent"]
    for i, c in enumerate(classes):
        out[f"prob_{c}"] = proba[:, i]
    return out


def evaluate_model(name: str, *, artifacts: str | Path | None = None, out: str | Path | None = None,
                   label: str = "primary", make_plots: bool = True, latency: bool = True) -> dict:
    model_dir = Path(artifacts) if artifacts else artifact_dir(name)
    model, card = load_trained(name, model_dir)
    cfg = model.cfg
    out = Path(out) if out else results_dir(name, label)
    labels, financial, help_intent = cfg["labels"], cfg["financial_intents"], cfg["safe_fallback_intent"]
    splits, fingerprint = load_splits(cfg)
    if fingerprint["sha256"] != card["dataset"]["sha256"]:
        raise RuntimeError(f"{name}: artifact was trained on dataset {card['dataset']['sha256'][:12]}, "
                           f"current is {fingerprint['sha256'][:12]}")
    staging = out.with_name(out.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    print(f"[evaluate] {name} ({label}) -> {rel(out)}", flush=True)

    proba = {k: model.predict_proba(df["text"].tolist()) for k, df in splits.items()}
    for k, p in proba.items():
        if not np.allclose(p.sum(1), 1.0, atol=1e-5):
            raise ValueError(f"{name}: probabilities on {k} do not sum to 1")

    # ---------------------------------------------------------------- classification metrics
    metrics = {}
    for split in ("train", "validation", "test"):
        df = splits.__getattribute__(split)
        pred = np.asarray(labels, dtype=object)[proba[split].argmax(1)]
        m = evaluate_predictions(df["intent"], pred, proba[split], labels, split, name, card["run_label"],
                                 core_labels=labels[:8])
        m["correct"] = int((pred == df["intent"].to_numpy()).sum())
        m["definition"] = ("raw classifier metrics: argmax over the nine intents, no confidence threshold; "
                           f"accuracy = correct / {len(df)} {split} commands")
        metrics[split] = m
        save_json(m, staging / f"metrics_{split}.json")
    test = metrics["test"]
    assert test["n_samples"] == 270 or cfg["smoke"], "primary test set must have 270 commands"

    report_rows = [{"intent": r["intent"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                    "support": r["support"]} for r in test["per_class"]]
    report_rows += [{"intent": "macro avg", "precision": test["macro_precision"], "recall": test["macro_recall"],
                     "f1": test["macro_f1"], "support": test["n_samples"]},
                    {"intent": "weighted avg", "precision": test["weighted_precision"], "recall": test["weighted_recall"],
                     "f1": test["weighted_f1"], "support": test["n_samples"]},
                    {"intent": "accuracy (raw)", "precision": None, "recall": None, "f1": test["accuracy"],
                     "support": test["n_samples"]}]
    pd.DataFrame(report_rows).to_csv(staging / "classification_report.csv", index=False)
    pd.DataFrame(test["confusion_matrix"], index=pd.Index(labels, name="true\\predicted"),
                 columns=labels).to_csv(staging / "confusion_matrix.csv")
    save_json({"split": "test", "method": "one-vs-rest, per class; macro = unweighted mean over the 9 intents",
               "macro_roc_auc_ovr": test["macro_roc_auc_ovr"], "weighted_roc_auc_ovr": test["weighted_roc_auc_ovr"],
               "per_class": {r["intent"]: r["roc_auc_ovr"] for r in test["per_class"]},
               "validation_macro_roc_auc_ovr": metrics["validation"]["macro_roc_auc_ovr"]}, staging / "roc_auc.json")

    # ---------------------------------------------------------------- thresholds
    tcfg = cfg["threshold"]
    if "validation" in card["trained_on"]:
        # Validation is training data for a refit, so calibrating on it would be contaminated. As in the teammates'
        # notebooks, reuse the threshold calibrated on validation with the train-only (primary) model.
        primary_meta = results_dir(name).joinpath("experiment_metadata.json")
        if not primary_meta.exists():
            raise FileNotFoundError(f"Evaluate the primary {name} model first: its threshold is reused for the refit")
        from ..utils import load_json
        cal = {**load_json(primary_meta)["threshold_calibration"],
               "calibration_split": "validation, using the train-only primary model (refit trains on validation)",
               "inherited_from": rel(primary_meta)}
    else:
        cal = calibrate_threshold(proba["validation"], tcfg["target_known_coverage"])
    default = cal["threshold"]
    sweep = threshold_sweep(splits.test["intent"], proba["test"], proba["unknown"], labels, tcfg["sweep"], default,
                            financial, help_intent, proba["validation"])
    none_row = sweep[sweep["threshold_label"] == "none"].iloc[0]
    assert abs(none_row["overall_correct_routing_rate"] - test["accuracy"]) < 1e-12, \
        "no-threshold correct-routing rate must equal raw test accuracy"
    sweep.to_csv(staging / "threshold_sweep.csv", index=False)
    dense = dense_sweep(splits.test["intent"], proba["test"], proba["unknown"], labels, financial, help_intent,
                        proba["validation"])
    dense.to_csv(staging / "threshold_sweep_dense.csv", index=False)

    _prediction_frame(splits.test, proba["test"], labels, default).to_csv(staging / "test_predictions.csv", index=False)
    _prediction_frame(splits.validation, proba["validation"], labels, default).to_csv(
        staging / "validation_predictions.csv", index=False)
    upf = _prediction_frame(splits.unknown, proba["unknown"], labels, default)
    upf["routed_to_financial_at_default"] = upf["accepted_at_default_threshold"] & upf["predicted_intent"].isin(financial)
    upf.to_csv(staging / "unknown_predictions.csv", index=False)

    # ---------------------------------------------------------------- latency, size, robustness
    lat = None
    if latency:
        lc = cfg["latency"]
        reps = 30 if cfg["smoke"] else lc["repeats"]
        summary, _ = measure_latency(lambda b: model.predict_proba(b), splits.test["text"].tolist(),
                                     batch_size=lc["batch_size"], warmup=min(lc["warmup"], reps), repeats=reps,
                                     seed=cfg["model"]["seed"])
        lat = {"scope": "single command, end to end (preprocessing/tokenization + model + softmax)",
               "device": getattr(getattr(model, "device", None), "type", None) or "cpu",
               "p50_ms": summary["median_ms"], "p95_ms": summary["p95_ms"], **summary,
               "hardware": {k: environment_info()[k] for k in ("cpu", "gpu", "cuda_available")}}
        save_json(lat, staging / "latency_metrics.json")

    rng = random.Random(cfg["robustness"]["typo_seed"])
    typo_texts = [add_typos(t, rng) for t in splits.test["text"]]
    tp = np.asarray(labels, dtype=object)[model.predict_proba(typo_texts).argmax(1)]
    from sklearn.metrics import f1_score
    robust = {"description": "one keyboard slip per test command (TF-IDF teammate's add_typos, seed "
                             f"{cfg['robustness']['typo_seed']}); secondary robustness check",
              "accuracy": float((tp == splits.test["intent"].to_numpy()).mean()),
              "macro_f1": float(f1_score(splits.test["intent"], tp, labels=labels, average="macro")),
              "clean_accuracy": test["accuracy"]}
    save_json(robust, staging / "robustness_typos.json")

    # ---------------------------------------------------------------- secondary analyses
    sec = cfg["secondary"]
    vidx = np.array([labels.index(i) for i in splits.validation["intent"]])
    tidx = np.array([labels.index(i) for i in splits.test["intent"]])
    menu_t = threshold_for_coverage(proba["validation"].max(1), sec["three_tier_menu_coverage"])
    secondary = {"label": "SECONDARY analyses - not part of the primary comparison",
                 "temperature_calibration": temperature_calibration(proba["validation"], vidx, proba["test"], tidx),
                 "three_tier_routing": {
                     "description": "Ricardo's three-tier gate re-derived without unknown labels: open threshold = "
                                    "primary default; menu threshold = validation threshold for "
                                    f"{sec['three_tier_menu_coverage']:.0%} coverage",
                     **three_tier_routing(splits.test["intent"], proba["test"], proba["unknown"], labels, default,
                                          min(menu_t, default), sec["menu_size"], financial)}}
    save_json(secondary, staging / "secondary_analyses.json")

    # ---------------------------------------------------------------- metadata
    files_sizes = card["size_bytes"]
    meta = {
        "schema_version": SCHEMA_VERSION, "model": name, "display_name": card["display_name"],
        "original_author": card["original_author"], "run_label": label, "protocol": card["protocol"],
        "variant": card["variant"], "smoke_test": card["smoke_test"], "primary_comparison": card["primary_comparison"],
        "trained_on": card["trained_on"], "selected_on": card["selected_on"],
        "evaluated_utc": utc_now(), "evaluation_command": command_line(), "training_command": card["command"],
        "git_commit": git_commit(), "dataset": fingerprint, "class_order": labels,
        "artifact": {"directory": rel(model_dir), "size_bytes": files_sizes, "size_mb": files_sizes / 1e6,
                     "files": card["files"], "model_card_sha256": sha256_file(model_dir / CARD)},
        "n_parameters": model.n_parameters(), "selection": card["selection"],
        "hyperparameters": card["selection"].get("params"),
        "training_seconds_final_fit": card["training_seconds_final_fit"],
        "training_wall_seconds_incl_search": card["training_wall_seconds_incl_search"],
        "threshold_calibration": cal,
        "headline": {"raw_test_accuracy": test["accuracy"], "raw_test_correct": test["correct"],
                     "raw_test_n": test["n_samples"], "test_macro_f1": test["macro_f1"],
                     "test_macro_precision": test["macro_precision"], "test_macro_recall": test["macro_recall"],
                     "test_weighted_f1": test["weighted_f1"], "test_macro_roc_auc_ovr": test["macro_roc_auc_ovr"],
                     "train_accuracy": metrics["train"]["accuracy"],
                     "validation_accuracy": metrics["validation"]["accuracy"],
                     "default_threshold": default,
                     "latency_p50_ms": lat["p50_ms"] if lat else None, "latency_p95_ms": lat["p95_ms"] if lat else None},
        "environment_training": card["environment"], "environment_evaluation": environment_info(),
    }
    save_json(meta, staging / "experiment_metadata.json")

    if make_plots:
        title = f"{card['display_name']} - test set (n = {test['n_samples']})"
        plots.confusion_matrix_png(test["confusion_matrix"], labels, f"Confusion matrix: {title}",
                                   staging / "confusion_matrix.png")
        plots.roc_curves_png(splits.test["intent"], proba["test"], labels, f"One-vs-rest ROC: {title}",
                             staging / "roc_curves.png")
        plots.threshold_sweep_png(dense, default, f"Threshold trade-off: {card['display_name']}",
                                  staging / "threshold_sweep.png")
    write_report(staging, meta, metrics, sweep, lat, robust, secondary)

    missing = [f for f in STANDARD_FILES if not (staging / f).exists() and not (f.endswith(".png") and not make_plots)
               and not (f == "latency_metrics.json" and not latency)]
    if missing:
        raise RuntimeError(f"{name}: standard result files missing: {missing}")
    publish_dir(staging, out)
    return meta
