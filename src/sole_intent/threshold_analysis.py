"""Exploratory confidence-threshold analysis on the frozen Baseline V1 model.

Reuses V1's saved class probabilities (``results/predictions/*_predictions.csv``), cross-checked
against the saved model, so nothing is retrained. Outputs go to ``results/threshold_analysis/`` only;
the analysis refuses to write into V1 output folders and verifies V1 artifact hashes before and after.

The test and unknown-evaluation sets were already inspected in earlier experiments. Everything here is
therefore exploratory and must not be used to claim that any threshold is statistically optimal.

Two layers are kept separate throughout:
- **model decision**: the top supported intent, or ``UNKNOWN_INTENT`` when max probability < threshold;
- **application action**: ``ROUTE_<INTENT>``, or ``FALLBACK_HELP_OR_CLARIFY`` for ``UNKNOWN_INTENT``.
A genuine ``HELP_SUPPORT`` classification routes to help; it is not a rejection.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from . import plots
from .data_loader import get_splits, load_dataset
from .preprocessing import canonical_key
from .rejection import UNKNOWN_INTENT, application_action, apply_rejection, max_confidence, threshold_sweep
from .train import load_bundle, reorder_proba
from .utils import (PROJECT_ROOT, load_config, load_json, resolve_path, save_json, sha256_file, utc_now)

DEFAULT_CONFIG = "config/threshold_analysis.yaml"
V1_ARTIFACT_DIRS = ["models", "results/metrics", "results/figures", "results/predictions", "results/reports"]
V1_ARTIFACT_FILES = ["config/baseline.yaml", "Sole_Smart_Actions_Intent_Training_v4_Cleaned.xlsx"]
HELP = "HELP_SUPPORT"

KNOWN_OUTCOMES = ["correct_accept", "incorrect_accept", "false_reject"]
UNKNOWN_OUTCOMES = ["correct_reject", "accepted_as_help_support", "accepted_as_financial", "accepted_as_other_intent"]


# ------------------------------------------------------------------ integrity
def v1_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """SHA-256 of every Baseline V1 artifact (models, outputs, config, source data)."""
    files = [p for d in V1_ARTIFACT_DIRS for p in sorted((root / d).rglob("*")) if p.is_file()]
    files += [root / f for f in V1_ARTIFACT_FILES if (root / f).exists()]
    return {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in files}


def assert_safe_output_dir(output_dir: Path, root: Path = PROJECT_ROOT) -> None:
    out = output_dir.resolve()
    for d in V1_ARTIFACT_DIRS:
        protected = (root / d).resolve()
        if out == protected or protected in out.parents:
            raise ValueError(f"Refusing to write threshold analysis into Baseline V1 folder: {protected}")


def threshold_tag(threshold: float) -> str:
    """0.35 -> '035' (used in output filenames)."""
    return f"{int(round(threshold * 100)):03d}"


# ------------------------------------------------------------------ data
def load_v1_probabilities(cfg: dict, labels: list[str], recompute: bool = False) -> dict:
    """Return frames + probability matrices for the known split, reference split and unknowns.
    Reads V1's saved probabilities; recomputes from the saved model only if they are unavailable."""
    pred_dir = resolve_path(cfg["predictions_dir"])
    prob_cols = [f"prob_{c}" for c in labels]
    names = {"known": cfg["known_split"], "reference": cfg["reference_known_split"], "unknown": "unknown"}
    paths = {k: pred_dir / f"{v}_predictions.csv" for k, v in names.items()}
    out = {"source": "saved_v1_predictions"}
    if not recompute and all(p.exists() for p in paths.values()):
        for key, path in paths.items():
            frame = pd.read_csv(path)
            missing = [c for c in prob_cols if c not in frame.columns]
            if missing:
                break
            out[key] = frame
            out[f"{key}_proba"] = frame[prob_cols].to_numpy(dtype=float)
        else:
            return out
    # Fallback: score the frozen model (still no retraining).
    base_cfg = load_config(cfg["baseline_config"])
    splits = get_splits(load_dataset(base_cfg), base_cfg).as_dict()
    bundle = load_bundle(resolve_path(cfg["model_path"]))
    out["source"] = "recomputed_from_saved_model"
    for key, split in names.items():
        frame = splits[split].rename(columns={"intent": "actual_intent"}).copy()
        out[key] = frame
        out[f"{key}_proba"] = reorder_proba(bundle["pipeline"], frame["text"], labels)
    return out


def verify_against_model(bundle: dict, data: dict, labels: list[str]) -> float:
    """Max |difference| between saved probabilities and the saved model's predict_proba."""
    diffs = [np.abs(reorder_proba(bundle["pipeline"], data[k]["text"], labels) - data[f"{k}_proba"]).max()
             for k in ("known", "reference", "unknown")]
    return float(max(diffs))


# ------------------------------------------------------------------ metrics
def _rate(num, den):
    return float(num / den) if den else None


def _macro_f1(y_true, y_pred, labels) -> tuple[float | None, int]:
    if len(y_true) == 0:
        return None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None,
                                                     zero_division=np.nan)
    defined = ~np.isnan(f)
    return (float(f[defined].mean()) if defined.any() else None), int(defined.sum())


def known_outcomes(y_true, proba, labels, threshold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(top intent, model decision, outcome) for legitimate commands."""
    y_true = np.asarray(y_true, dtype=object)
    top = np.asarray(labels, dtype=object)[np.asarray(proba).argmax(axis=1)]
    decision = apply_rejection(proba, labels, threshold)
    outcome = np.where(decision == UNKNOWN_INTENT, "false_reject",
                       np.where(decision == y_true, "correct_accept", "incorrect_accept")).astype(object)
    return top, decision, outcome


def unknown_outcomes(proba, labels, threshold, financial) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(top intent, model decision, outcome) for unsupported commands."""
    top = np.asarray(labels, dtype=object)[np.asarray(proba).argmax(axis=1)]
    decision = apply_rejection(proba, labels, threshold)
    outcome = np.select(
        [decision == UNKNOWN_INTENT, decision == HELP, np.isin(decision, financial)],
        ["correct_reject", "accepted_as_help_support", "accepted_as_financial"],
        default="accepted_as_other_intent").astype(object)
    return top, decision, outcome


def threshold_metrics(known_true, known_proba, unknown_proba, labels: list[str], threshold: float,
                      financial: list[str], unknown_groups=None) -> dict:
    """All per-threshold metrics. Model-classification metrics and application-level fallback metrics
    are reported under separate keys."""
    known_true = np.asarray(known_true, dtype=object)
    _, kd, ko = known_outcomes(known_true, known_proba, labels, threshold)
    _, ud, uo = unknown_outcomes(unknown_proba, labels, threshold, financial)
    nk, nu = len(known_true), len(ud)
    accepted = ko != "false_reject"
    acc_f1, acc_f1_classes = _macro_f1(known_true[accepted], kd[accepted], labels)
    n_reject_u = int((uo == "correct_reject").sum())
    n_help_u = int((uo == "accepted_as_help_support").sum())
    fin = {f: int((ud == f).sum()) for f in financial}
    per_intent = {}
    for lab in labels:
        m = known_true == lab
        rej = int((m & (ko == "false_reject")).sum())
        per_intent[lab] = {"support": int(m.sum()), "false_rejected": rej, "false_rejection_rate": _rate(rej, int(m.sum()))}
    known_to_fin = int((accepted & (kd != known_true) & np.isin(kd, financial)).sum())
    out = {
        "threshold": float(threshold),
        "model_classification": {
            "known": {
                "n": nk,
                "correct_accept": int((ko == "correct_accept").sum()),
                "incorrect_accept": int((ko == "incorrect_accept").sum()),
                "false_reject": int((ko == "false_reject").sum()),
                "coverage": _rate(int(accepted.sum()), nk),
                "false_rejection_rate": _rate(int((~accepted).sum()), nk),
                "accepted_accuracy": _rate(int((ko == "correct_accept").sum()), int(accepted.sum())),
                "accepted_macro_f1": acc_f1,
                "accepted_macro_f1_defined_classes": acc_f1_classes,
                "end_to_end_accuracy": _rate(int((ko == "correct_accept").sum()), nk),
                "misrouted_to_financial": known_to_fin,
                "false_rejection_by_intent": per_intent,
            },
            "unknown": {
                "n": nu,
                "rejected_as_unknown_intent": n_reject_u,
                "rejection_rate": _rate(n_reject_u, nu),
                "false_accept_rate": _rate(nu - n_reject_u, nu),
                "classified_as_help_support": n_help_u,
                "classified_as_help_support_rate": _rate(n_help_u, nu),
                "classified_as_financial": int(sum(fin.values())),
                "classified_as_financial_rate": _rate(int(sum(fin.values())), nu),
                "classified_as_financial_by_intent": fin,
                "classified_as_other_intent": int((uo == "accepted_as_other_intent").sum()),
            },
        },
        "application": {
            "unknown_fallback_triggered": n_reject_u,
            "unknown_routed_to_help_support": n_help_u,
            "unknown_safe_fallback": n_reject_u + n_help_u,
            "unknown_safe_fallback_rate": _rate(n_reject_u + n_help_u, nu),
            "unknown_routed_to_financial_action": int(sum(fin.values())),
            "known_fallback_triggered": int((~accepted).sum()),
            "note": ("Safe fallback = unknown command either rejected (app shows HELP/clarify fallback) or "
                     "classified HELP_SUPPORT by the model (app routes to help). The second case is a model "
                     "misclassification with a benign app outcome, not a rejection."),
        },
    }
    if unknown_groups is not None:
        rejected = pd.Series(ud == UNKNOWN_INTENT).groupby(np.asarray(unknown_groups)).mean()
        out["model_classification"]["unknown"]["rejection_rate_unique_phrases"] = float(rejected.mean())
    return out


def flatten(m: dict, labels: list[str], financial: list[str]) -> dict:
    k, u, a = m["model_classification"]["known"], m["model_classification"]["unknown"], m["application"]
    row = {
        "threshold": m["threshold"],
        "known_n": k["n"], "known_correct_accept": k["correct_accept"], "known_incorrect_accept": k["incorrect_accept"],
        "known_false_reject": k["false_reject"], "known_coverage": k["coverage"],
        "known_false_rejection_rate": k["false_rejection_rate"], "accepted_accuracy": k["accepted_accuracy"],
        "accepted_macro_f1": k["accepted_macro_f1"], "end_to_end_known_accuracy": k["end_to_end_accuracy"],
        "known_misrouted_to_financial": k["misrouted_to_financial"],
        "unknown_n": u["n"], "unknown_rejected": u["rejected_as_unknown_intent"],
        "unknown_rejection_rate": u["rejection_rate"], "unknown_false_accept_rate": u["false_accept_rate"],
        "unknown_rejection_rate_unique_phrases": u.get("rejection_rate_unique_phrases"),
        "unknown_to_help_support": u["classified_as_help_support"],
        "unknown_to_help_support_rate": u["classified_as_help_support_rate"],
        "safe_fallback_rate": a["unknown_safe_fallback_rate"],
        "unknown_to_financial": u["classified_as_financial"],
        "unknown_to_financial_rate": u["classified_as_financial_rate"],
        **{f"unknown_to_{f}": u["classified_as_financial_by_intent"][f] for f in financial},
        "unknown_to_other_intents": u["classified_as_other_intent"],
    }
    for lab in labels:
        row[f"false_rejected_{lab}"] = k["false_rejection_by_intent"][lab]["false_rejected"]
    return row


# ------------------------------------------------------------------ per-row export
def decision_table(data: dict, labels, threshold, financial) -> pd.DataFrame:
    rows = []
    for key, dataset in (("known", "legitimate_test"), ("unknown", "unknown_evaluation")):
        frame, proba = data[key], data[f"{key}_proba"]
        if key == "known":
            top, dec, oc = known_outcomes(frame["actual_intent"], proba, labels, threshold)
        else:
            top, dec, oc = unknown_outcomes(proba, labels, threshold, financial)
        part = pd.DataFrame({
            "dataset": dataset, "row_id": frame["row_id"].to_numpy(), "excel_row": frame["excel_row"].to_numpy(),
            "text": frame["text"].to_numpy(), "actual_intent": frame["actual_intent"].to_numpy(),
            "top_intent": top, "max_confidence": max_confidence(proba), "threshold": threshold,
            "model_decision": dec, "application_action": [application_action(d) for d in dec], "outcome": oc,
        })
        rows.append(pd.concat([part, pd.DataFrame(proba, columns=[f"prob_{c}" for c in labels])], axis=1))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ orchestration
def resolve_thresholds(values, baseline_threshold: float) -> list[float]:
    out = []
    for v in values:
        out.append(float(baseline_threshold) if str(v).lower() == "baseline" else float(v))
    return sorted(set(out))


def run_threshold_analysis(config_path=None, thresholds=None, candidate: float | None = None,
                           output_dir=None, recompute: bool = False, make_plots: bool = True) -> dict:
    cfg = load_config(config_path or DEFAULT_CONFIG)
    base_cfg = load_config(cfg["baseline_config"])
    labels = base_cfg["labels"]
    financial = cfg["financial_intents"]
    out_dir = resolve_path(output_dir or cfg["output_dir"])
    assert_safe_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_before = v1_manifest()
    model_path = resolve_path(cfg["model_path"])
    bundle = load_bundle(model_path)
    if bundle["class_order"] != labels:
        raise ValueError("Model class order differs from config labels")
    baseline_thr = float(bundle["rejection_threshold"])
    grid = resolve_thresholds(thresholds if thresholds is not None else cfg["thresholds"], baseline_thr)
    candidate = float(candidate if candidate is not None else cfg["candidate_threshold"])
    if candidate not in grid:
        grid = sorted(grid + [candidate])

    data = load_v1_probabilities(cfg, labels, recompute=recompute)
    max_diff = verify_against_model(bundle, data, labels)
    if max_diff > 1e-9:
        raise ValueError(f"Saved V1 probabilities disagree with the saved model (max diff {max_diff:.2e})")
    known_true = data["known"]["actual_intent"].to_numpy(dtype=object)
    groups = data["unknown"]["text"].map(canonical_key).to_numpy()

    def metrics_at(t):
        return threshold_metrics(known_true, data["known_proba"], data["unknown_proba"], labels, t, financial, groups)

    # Reproduce published V1 numbers at the frozen threshold before anything else.
    v1_published = load_json(resolve_path(cfg["baseline_rejection_metrics"]))["primary_frozen_threshold_test_plus_unknown"]
    m0 = metrics_at(baseline_thr)
    k0, u0 = m0["model_classification"]["known"], m0["model_classification"]["unknown"]
    reproduced = {
        "correct_supported_classification": k0["correct_accept"],
        "incorrect_supported_classification": k0["incorrect_accept"],
        "false_rejection_of_supported": k0["false_reject"],
        "correct_rejection_of_unsupported": u0["rejected_as_unknown_intent"],
        "false_acceptance_of_unsupported": u0["n"] - u0["rejected_as_unknown_intent"],
    }
    if reproduced != v1_published["counts"]:
        raise ValueError(f"Could not reproduce V1 counts: {reproduced} vs {v1_published['counts']}")

    results = [metrics_at(t) for t in grid]
    ref_true = data["reference"]["actual_intent"].to_numpy(dtype=object)
    for r in results:
        acc = ~(apply_rejection(data["reference_proba"], labels, r["threshold"]) == UNKNOWN_INTENT)
        r["reference_validation_known_coverage"] = float(acc.mean())
    table = pd.DataFrame([flatten(r, labels, financial) for r in results])
    table.insert(1, "label", [("0.196 (V1 frozen)" if t == baseline_thr else f"{t:.2f}") +
                               (" [candidate]" if t == candidate else "") for t in table["threshold"]])
    table.insert(2, "is_baseline_v1", table["threshold"] == baseline_thr)
    table.insert(3, "is_candidate", table["threshold"] == candidate)
    table["validation_known_coverage_reference"] = [r["reference_validation_known_coverage"] for r in results]
    table.to_csv(out_dir / "threshold_comparison.csv", index=False)

    by_intent = pd.DataFrame([{"threshold": r["threshold"], "intent": lab, **v}
                              for r in results for lab, v in r["model_classification"]["known"]["false_rejection_by_intent"].items()])
    by_intent.to_csv(out_dir / "legitimate_rejection_by_intent.csv", index=False)

    dense_t = np.round(np.arange(0, 1 + 1e-9, cfg["dense_sweep_step"]), 4)
    dense = threshold_sweep(known_true, data["known_proba"], data["unknown_proba"], labels, dense_t)
    dense.to_csv(out_dir / "dense_threshold_sweep.csv", index=False)

    # Candidate detail
    tag = threshold_tag(candidate)
    detail = decision_table(data, labels, candidate, financial)
    detail.to_csv(out_dir / f"threshold_{tag}_predictions.csv", index=False)
    errors = detail[~detail["outcome"].isin(["correct_accept", "correct_reject"])].sort_values(
        ["dataset", "outcome", "max_confidence"], ascending=[True, True, False])
    errors.to_csv(out_dir / f"threshold_{tag}_errors.csv", index=False)
    base_detail = decision_table(data, labels, baseline_thr, financial)
    changes = detail[["dataset", "row_id", "text", "actual_intent", "top_intent", "max_confidence"]].copy()
    changes["outcome_v1"] = base_detail["outcome"].to_numpy()
    changes[f"outcome_{tag}"] = detail["outcome"].to_numpy()
    changes = changes[changes["outcome_v1"] != changes[f"outcome_{tag}"]]
    changes.to_csv(out_dir / f"threshold_{tag}_vs_v1_changes.csv", index=False)

    payload = {
        "generated_utc": utc_now(),
        "status": "EXPLORATORY - test and unknown-evaluation sets were inspected in earlier experiments",
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256_file(model_path),
        "probability_source": data["source"],
        "max_abs_diff_saved_vs_model_probabilities": max_diff,
        "baseline_v1_threshold": baseline_thr,
        "baseline_v1_counts_reproduced": True,
        "candidate_threshold": candidate,
        "thresholds": grid,
        "known_split": cfg["known_split"],
        "decision_rule": "max_probability < threshold -> UNKNOWN_INTENT; else top supported intent",
        "application_mapping": {"UNKNOWN_INTENT": "FALLBACK_HELP_OR_CLARIFY (offers HELP_SUPPORT or clarification)",
                                "<INTENT>": "ROUTE_<INTENT>"},
        "results": results,
    }
    save_json(payload, out_dir / "threshold_comparison.json")

    if make_plots:
        plots.apply_style(base_cfg["plots"]["dpi"])
        plots.threshold_panels(table, baseline_thr, candidate, out_dir / "threshold_comparison.png")
        plots.coverage_vs_rejection(table, dense, baseline_thr, candidate, out_dir / "coverage_vs_rejection.png")
        plots.safe_fallback_comparison(table, out_dir / "safe_fallback_comparison.png")
        plots.financial_routing_risk(table, financial, out_dir / "financial_routing_risk.png")
        plots.legitimate_rejection_heatmap(by_intent, labels, out_dir / "legitimate_rejection_by_intent.png")

    from .threshold_report import write_threshold_reports
    ctx = {"cfg": cfg, "labels": labels, "financial": financial, "table": table, "results": results,
           "baseline_thr": baseline_thr, "candidate": candidate, "tag": tag, "detail": detail, "errors": errors,
           "changes": changes, "payload": payload, "out_dir": out_dir, "by_intent": by_intent}
    write_threshold_reports(ctx)

    manifest_after = v1_manifest()
    changed = sorted(k for k in set(manifest_before) | set(manifest_after)
                     if manifest_before.get(k) != manifest_after.get(k))
    save_json({"generated_utc": utc_now(), "baseline_v1_unchanged": not changed, "changed_files": changed,
               "n_files": len(manifest_after), "sha256": manifest_after}, out_dir / "baseline_v1_manifest.json")
    if changed:
        raise RuntimeError(f"Baseline V1 artifacts changed during analysis: {changed}")
    ctx["v1_unchanged"] = True
    return ctx
