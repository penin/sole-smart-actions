"""Model-agnostic evaluation protocol shared by all Sole Smart Actions experiments.

Any model (BoW, CNN, LSTM, transformer) can be evaluated with :func:`evaluate_predictions`
given true labels, predicted labels, a probability matrix whose columns follow the canonical
``class_order`` from ``config/history/bow_v4.yaml``, and identifying metadata.

Undefined metrics (e.g. ROC-AUC for a class with no positives, precision for a class that is
never predicted) are reported as ``None`` with a reason in ``undefined_metrics``; they are
never silently replaced by zero.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)

SUMMARY_FIELDS = [
    "model_name", "experiment_id", "variant", "dataset_version", "dataset_sha256", "config_sha256",
    "random_seed", "train_accuracy", "validation_accuracy", "test_accuracy",
    "train_macro_f1", "validation_macro_f1", "test_macro_f1", "test_weighted_f1",
    "test_macro_roc_auc", "test_log_loss", "test_balanced_accuracy", "test_core8_macro_f1",
    "test_leakage_controlled_macro_f1", "rejection_threshold", "known_intent_coverage",
    "unknown_rejection_rate", "unknown_false_accept_rate", "inference_p50_ms", "inference_p95_ms",
    "inference_p99_ms", "throughput_per_s", "training_seconds", "model_size_mb", "n_parameters",
    "vocabulary_size", "notes",
]


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def expected_calibration_error(y_true, y_pred, confidence, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Top-label ECE with equal-width bins, plus the per-bin table behind a reliability diagram."""
    correct = (np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object)).astype(float)
    confidence = np.asarray(confidence, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else np.nan
        conf = float(confidence[mask].mean()) if n else np.nan
        if n:
            ece += n / len(confidence) * abs(acc - conf)
        rows.append({"bin_lower": lo, "bin_upper": hi, "count": n, "mean_confidence": conf, "accuracy": acc})
    return float(ece), pd.DataFrame(rows)


def multiclass_log_loss(y_true, proba, class_order: list[str]) -> float:
    """sklearn's log_loss assumes columns in sorted label order; reorder from canonical order first."""
    sorted_idx = np.argsort(np.asarray(class_order, dtype=object))
    return float(log_loss(y_true, np.asarray(proba)[:, sorted_idx], labels=sorted(class_order)))


def _classification_block(y_true, y_pred, labels, undefined: dict, prefix: str = "") -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                     average=None, zero_division=np.nan)
    out = {}
    for name, values in (("precision", p), ("recall", r), ("f1", f)):
        values = np.asarray(values, dtype=float)
        bad = [labels[i] for i in np.where(np.isnan(values))[0]]
        if bad:
            undefined[f"{prefix}{name}"] = f"undefined for {bad}; macro/weighted averaged over defined classes only"
        defined = ~np.isnan(values)
        out[f"macro_{name}"] = _nan_to_none(values[defined].mean()) if defined.any() else None
        w = s[defined]
        out[f"weighted_{name}"] = _nan_to_none(np.average(values[defined], weights=w)) if w.sum() else None
    return out, p, r, f, s


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str], y_proba,
                         class_order: list[str], split_name: str, model_name: str,
                         experiment_id: str, core_labels: list[str] | None = None) -> dict:
    """Compute the standard metric set for one split. Returns a JSON-serializable dict."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    proba = np.asarray(y_proba, dtype=float)
    n, k = len(y_true), len(class_order)
    if proba.shape != (n, k):
        raise ValueError(f"y_proba shape {proba.shape} != ({n}, {k})")
    unexpected = set(y_true) - set(class_order)
    if unexpected:
        raise ValueError(f"y_true contains labels outside class_order: {unexpected}")

    undefined: dict[str, str] = {}
    result = {"model_name": model_name, "experiment_id": experiment_id, "split": split_name,
              "n_samples": n, "class_order": list(class_order)}
    result["accuracy"] = float(accuracy_score(y_true, y_pred))

    block, p, r, f, s = _classification_block(y_true, y_pred, class_order, undefined)
    result.update(block)
    present = s > 0
    result["balanced_accuracy"] = _nan_to_none(np.asarray(r, dtype=float)[present].mean()) if present.any() else None
    if not present.all():
        undefined["balanced_accuracy"] = "computed over classes present in y_true only"

    result["log_loss"] = multiclass_log_loss(y_true, proba, class_order)

    # One-vs-rest ROC-AUC and average precision, per class
    y_true_idx = np.array([class_order.index(t) for t in y_true])
    auc, ap = np.full(k, np.nan), np.full(k, np.nan)
    for i, label in enumerate(class_order):
        positives = y_true_idx == i
        if positives.all() or not positives.any():
            undefined[f"roc_auc[{label}]"] = "needs both positive and negative examples"
            undefined[f"average_precision[{label}]"] = "no positive examples" if not positives.any() else "no negatives"
            continue
        auc[i] = roc_auc_score(positives, proba[:, i])
        ap[i] = average_precision_score(positives, proba[:, i])
    defined = ~np.isnan(auc)
    result["macro_roc_auc_ovr"] = _nan_to_none(auc[defined].mean()) if defined.any() else None
    result["weighted_roc_auc_ovr"] = (_nan_to_none(np.average(auc[defined], weights=s[defined]))
                                      if defined.any() and s[defined].sum() else None)
    result["macro_average_precision"] = _nan_to_none(ap[defined].mean()) if defined.any() else None

    ranks = np.argsort(-proba, axis=1)
    for k_top in (2, 3):
        if k_top < k:
            result[f"top{k_top}_accuracy"] = float(np.mean([y_true_idx[i] in ranks[i, :k_top] for i in range(n)]))
    conf = proba.max(axis=1)
    result["mean_confidence"] = float(conf.mean())
    result["expected_calibration_error"], _ = expected_calibration_error(y_true, y_pred, conf)

    result["per_class"] = [
        {"intent": label, "precision": _nan_to_none(p[i]), "recall": _nan_to_none(r[i]),
         "f1": _nan_to_none(f[i]), "support": int(s[i]), "roc_auc_ovr": _nan_to_none(auc[i]),
         "average_precision": _nan_to_none(ap[i])}
        for i, label in enumerate(class_order)
    ]
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    result["confusion_matrix"] = cm.tolist()
    result["confusion_matrix_normalized"] = [[_nan_to_none(v) for v in row] for row in cm_norm]

    if core_labels:
        result["core8_breakdown"] = core_breakdown(y_true, y_pred, core_labels)
    result["undefined_metrics"] = undefined
    return result


def core_breakdown(y_true, y_pred, core_labels: list[str]) -> dict:
    """Metrics restricted to rows whose true label is a core intent. Predictions outside the
    core set (e.g. CHECK_SPENDING) count as errors."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    mask = np.isin(y_true, core_labels)
    if not mask.any():
        return {"n_samples": 0}
    undefined: dict[str, str] = {}
    block, *_ = _classification_block(y_true[mask], y_pred[mask], core_labels, undefined, prefix="core8_")
    return {"n_samples": int(mask.sum()),
            "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
            **block,
            "predicted_outside_core": int((~np.isin(y_pred[mask], core_labels)).sum()),
            "undefined_metrics": undefined}


def per_class_frame(metrics: dict) -> pd.DataFrame:
    frame = pd.DataFrame(metrics["per_class"])
    frame.insert(0, "split", metrics["split"])
    frame.insert(0, "model_name", metrics["model_name"])
    return frame


def confusion_frames(metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = metrics["class_order"]
    idx = pd.Index(labels, name="actual\\predicted")
    return (pd.DataFrame(metrics["confusion_matrix"], index=idx, columns=labels),
            pd.DataFrame(metrics["confusion_matrix_normalized"], index=idx, columns=labels))


def measure_latency(predict_fn: Callable[[list[str]], object], texts: Sequence[str], *,
                    batch_size: int = 1, warmup: int = 50, repeats: int = 1000,
                    seed: int = 42) -> tuple[dict, np.ndarray]:
    """Wall-clock latency of ``predict_fn`` on batches drawn from ``texts``.

    Warm-up calls are discarded. Each measured call uses ``time.perf_counter`` around a single
    end-to-end call (preprocessing + vectorization + inference). Returns summary and raw ms."""
    rng = np.random.default_rng(seed)
    texts = list(texts)

    def batch():
        if batch_size >= len(texts):
            return texts[:batch_size]
        return [texts[i] for i in rng.choice(len(texts), size=batch_size, replace=False)]

    for _ in range(warmup):
        predict_fn(batch())
    samples = np.empty(repeats)
    for i in range(repeats):
        b = batch()
        start = time.perf_counter()
        predict_fn(b)
        samples[i] = (time.perf_counter() - start) * 1000
    actual_batch = min(batch_size, len(texts))
    summary = {
        "batch_size": actual_batch, "warmup_calls": warmup, "measured_calls": repeats,
        "mean_ms": float(samples.mean()), "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)), "p99_ms": float(np.percentile(samples, 99)),
        "min_ms": float(samples.min()), "max_ms": float(samples.max()),
        "throughput_predictions_per_s": float(actual_batch / (np.median(samples) / 1000)),
    }
    return summary, samples


def _macro_f1(y_true, y_pred, labels) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None,
                                                     zero_division=np.nan)
    return float(np.nanmean(f)) if not np.all(np.isnan(f)) else np.nan


def cluster_bootstrap(y_true, predictions: dict[str, Sequence[str]], class_order: list[str],
                      groups, n_resamples: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile confidence intervals for accuracy and macro F1, resampling whole groups
    (e.g. identical normalized phrases) so repeated utterances do not make intervals artificially
    narrow. With two or more models, also reports paired differences against the first model."""
    y_true = np.asarray(y_true, dtype=object)
    preds = {k: np.asarray(v, dtype=object) for k, v in predictions.items()}
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = [np.where(groups == g)[0] for g in unique]
    rng = np.random.default_rng(seed)
    draws = {k: {"accuracy": [], "macro_f1": []} for k in preds}
    for _ in range(n_resamples):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        for k, p in preds.items():
            draws[k]["accuracy"].append(float((y_true[idx] == p[idx]).mean()))
            draws[k]["macro_f1"].append(_macro_f1(y_true[idx], p[idx], class_order))

    def interval(values, point):
        values = np.asarray(values, dtype=float)
        return {"estimate": point, "ci_low": float(np.nanpercentile(values, 100 * alpha / 2)),
                "ci_high": float(np.nanpercentile(values, 100 * (1 - alpha / 2)))}

    out = {"method": f"cluster bootstrap over {len(unique)} groups, {n_resamples} resamples, "
                     f"{int(100 * (1 - alpha))}% percentile intervals", "models": {}}
    for k, p in preds.items():
        out["models"][k] = {"accuracy": interval(draws[k]["accuracy"], float((y_true == p).mean())),
                            "macro_f1": interval(draws[k]["macro_f1"], _macro_f1(y_true, p, class_order))}
    names = list(preds)
    for other in names[1:]:
        diff = np.asarray(draws[names[0]]["macro_f1"]) - np.asarray(draws[other]["macro_f1"])
        point = out["models"][names[0]]["macro_f1"]["estimate"] - out["models"][other]["macro_f1"]["estimate"]
        out[f"macro_f1_difference_{names[0]}_minus_{other}"] = interval(diff, point) | {
            "share_of_resamples_positive": float(np.mean(diff > 0))}
    return out


def summary_row(**fields) -> dict:
    """One comparable experiment_summary.csv row; unknown fields are rejected, missing ones are None."""
    extra = set(fields) - set(SUMMARY_FIELDS)
    if extra:
        raise ValueError(f"Unknown summary fields: {extra}")
    return {f: fields.get(f) for f in SUMMARY_FIELDS}
