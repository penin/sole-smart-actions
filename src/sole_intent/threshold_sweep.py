"""Versioned confidence-threshold sweep for a freshly trained model (e.g. the Dataset V5 run).

Reuses the tested building blocks of ``threshold_analysis`` (per-threshold metrics, per-row decisions) and the
threshold charts in ``plots``. Raw classifier accuracy is reported elsewhere; everything here is about the
rejection rule ``max probability < threshold -> UNKNOWN_INTENT``.

Only the validation-calibrated default is a methodological choice. Every other threshold is exploratory:
the test and unknown sets are scored at each one, so none can be chosen as "optimal" from these results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import plots
from .preprocessing import canonical_key
from .rejection import UNKNOWN_INTENT, apply_rejection, threshold_sweep
from .threshold_analysis import decision_table, flatten, threshold_metrics
from .utils import OutputPaths, save_json, utc_now


def _resolve(values, default: float) -> list[tuple[float, str]]:
    out = []
    for v in values:
        key = str(v).lower()
        if key == "none":
            out.append((0.0, "No threshold"))
        elif key == "default":
            out.append((float(default), f"Default {100 * default:.2f}% (validation-calibrated)"))
        else:
            out.append((float(v), f"{100 * float(v):.0f}%"))
    seen, unique = set(), []
    for t, label in sorted(out, key=lambda x: x[0]):
        if t not in seen:
            seen.add(t)
            unique.append((t, label))
    return unique


def run_threshold_sweep(cfg: dict, P: OutputPaths, labels: list[str], *, default_threshold: float,
                        test: pd.DataFrame, unknown: pd.DataFrame, test_proba: np.ndarray,
                        unknown_proba: np.ndarray, validation_proba: np.ndarray, financial: list[str]) -> dict:
    sw = cfg["threshold_sweep"]
    out_dir = P.results / "threshold_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = _resolve(sw["thresholds"], default_threshold)
    known_true = test["actual_intent"].to_numpy(dtype=object)
    groups = unknown["text"].map(canonical_key).to_numpy()

    results, rows = [], []
    for t, label in grid:
        m = threshold_metrics(known_true, test_proba, unknown_proba, labels, t, financial, groups)
        val_cov = float((apply_rejection(validation_proba, labels, t) != UNKNOWN_INTENT).mean())
        m["label"], m["validation_known_coverage"] = label, val_cov
        results.append(m)
        row = flatten(m, labels, financial)
        row = {"label": label, "is_default": t == default_threshold, "is_no_threshold": t == 0.0,
               **row, "known_accepted": row["known_n"] - row["known_false_reject"],
               "overall_correct_routing_rate": row["end_to_end_known_accuracy"],
               "validation_known_coverage": val_cov}
        rows.append(row)
    table = pd.DataFrame(rows)
    # Raw accuracy (no threshold) must equal the classifier's accuracy: a hard consistency check.
    raw = table[table["is_no_threshold"]]
    if len(raw) and not np.isclose(raw["overall_correct_routing_rate"].iloc[0], test["correct"].mean()):
        raise AssertionError("No-threshold routing rate differs from raw test accuracy")
    table.to_csv(out_dir / "threshold_sweep.csv", index=False)
    save_json({"generated_utc": utc_now(), "default_threshold": default_threshold,
               "default_selection": "validation known-intent coverage target "
                                    f"{cfg['rejection']['target_known_coverage']:.0%} (unknown labels not used)",
               "status": "exploratory except the validation-calibrated default",
               "known_set": f"test ({len(test)} legitimate commands)", "unknown_set": f"{len(unknown)} evaluation-only",
               "results": results}, out_dir / "threshold_sweep.json")

    by_intent = pd.DataFrame([{"threshold": r["threshold"], "intent": lab, **v}
                              for r in results for lab, v in r["model_classification"]["known"]["false_rejection_by_intent"].items()])
    by_intent.to_csv(out_dir / "legitimate_rejection_by_intent.csv", index=False)
    dense_t = np.round(np.arange(0, 1 + 1e-9, 0.01), 4)
    dense = threshold_sweep(known_true, test_proba, unknown_proba, labels, dense_t)
    dense.to_csv(out_dir / "dense_threshold_sweep.csv", index=False)

    data = {"known": test, "known_proba": test_proba,
            "unknown": unknown, "unknown_proba": unknown_proba}
    decision_table(data, labels, default_threshold, financial).to_csv(
        out_dir / "decisions_at_default_threshold.csv", index=False)

    chart = table
    tag = f"default {100 * default_threshold:.2f}%"
    plots.threshold_panels(chart, default_threshold, None, out_dir / "threshold_comparison.png",
                           ref_name=f"validation-calibrated default {100 * default_threshold:.2f}%",
                           title=f"Threshold comparison: {cfg['experiment']['experiment_id']} "
                                 "(exploratory; test + evaluation-only unknowns)")
    plots.coverage_vs_rejection(chart, dense, default_threshold, None, out_dir / "coverage_vs_rejection.png",
                                ref_short=tag)
    plots.safe_fallback_comparison(chart, out_dir / "safe_fallback_comparison.png")
    plots.financial_routing_risk(chart, financial, out_dir / "financial_routing_risk.png")
    plots.legitimate_rejection_heatmap(by_intent, labels, out_dir / "legitimate_rejection_by_intent.png")
    return {"table": table, "results": results, "by_intent": by_intent, "dir": out_dir,
            "default_threshold": default_threshold}
