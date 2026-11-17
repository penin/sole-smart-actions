"""Per-model ``evaluation_report.md``, generated from the computed metrics (no hard-coded results)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def pct(x, d=1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def num(x, d=3) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"


def write_report(directory: Path, meta: dict, metrics: dict, sweep: pd.DataFrame, lat: dict | None, robust: dict,
                 secondary: dict) -> Path:
    h, cal, t = meta["headline"], meta["threshold_calibration"], metrics["test"]
    banner = []
    if meta["smoke_test"]:
        banner.append("> **SMOKE TEST** - tiny grid / subset for pipeline checking only. These numbers are NOT "
                      "published results.")
    if not meta["primary_comparison"] and not meta["smoke_test"]:
        banner.append(f"> **SECONDARY EXPERIMENT ({meta['run_label']})** - trained on {', '.join(meta['trained_on'])}; "
                      "not part of the primary four-model comparison.")
    lines = [f"# {meta['display_name']} - evaluation report", "", *banner, "" if banner else "",
             f"_Generated {meta['evaluated_utc']} from the model's actual predictions. Dataset "
             f"`{meta['dataset']['version']}` (SHA-256 `{meta['dataset']['sha256'][:16]}...`). "
             f"Original implementation: {meta['original_author']}._", "",
             "## Headline (raw classifier, no confidence threshold)", "",
             "| Metric | Value |", "|---|---|",
             f"| Raw test accuracy | **{pct(h['raw_test_accuracy'])}** ({h['raw_test_correct']} / {h['raw_test_n']}) |",
             f"| Test macro precision / recall / F1 | {num(t['macro_precision'])} / {num(t['macro_recall'])} / "
             f"**{num(t['macro_f1'])}** |",
             f"| Test weighted F1 | {num(t['weighted_f1'])} |",
             f"| Test macro ROC-AUC (one-vs-rest) | {num(t['macro_roc_auc_ovr'])} |",
             f"| Train / validation accuracy | {pct(h['train_accuracy'])} / {pct(h['validation_accuracy'])} |",
             f"| Parameters | {meta['n_parameters']:,} |" if meta["n_parameters"] else "| Parameters | n/a |",
             f"| Serialized model size | {meta['artifact']['size_mb']:.2f} MB |",
             f"| Training time (final fit / incl. search) | {meta['training_seconds_final_fit']:.1f} s / "
             f"{meta['training_wall_seconds_incl_search']:.1f} s |"]
    if lat:
        lines.append(f"| Latency P50 / P95 (single command, {lat['device']}) | {lat['p50_ms']:.2f} / {lat['p95_ms']:.2f} ms |")
    lines += ["", "Raw accuracy counts every one of the legitimate test commands, before any confidence threshold.", "",
              "## Selected configuration (chosen on validation only)", "",
              f"- Hyperparameters: `{meta['hyperparameters']}`",
              f"- Rule: {meta['selection'].get('rule', 'n/a')}",
              f"- Configurations searched: {meta['selection'].get('n_configurations', 'n/a')}", "",
              "## Per-intent results (test)", "", "| Intent | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    lines += [f"| {r['intent']} | {num(r['precision'])} | {num(r['recall'])} | {num(r['f1'])} | {r['support']} |"
              for r in t["per_class"]]
    lines += ["", "## Confidence threshold", "",
              f"Default threshold **{cal['threshold']:.4f}**, calibrated on validation only: {cal['algorithm']}; "
              f"target coverage {pct(cal['target_known_coverage'], 0)}, actual validation coverage "
              f"{pct(cal['actual_validation_coverage'])}. No test or unknown-evaluation labels were used.", "",
              "| Threshold | Coverage | Accepted | Correct accepted | Accuracy among accepted | Overall correct routing "
              "| Rejected | Unknown rejected | Unknown -> HELP_SUPPORT | Unknown safe fallback | Unknown -> money flow |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sweep.iterrows():
        lab = f"default ({r['threshold']:.3f})" if r["threshold_label"] == "default" else r["threshold_label"]
        lines.append(f"| {lab} | {pct(r['known_coverage'])} | {r['accepted_known']} | {r['accepted_known_correct']} | "
                     f"{pct(r['accuracy_among_accepted'])} | {pct(r['overall_correct_routing_rate'])} | "
                     f"{r['rejected_known']} | {pct(r['unknown_rejection_rate'])} | {r['unknown_as_help_support']} | "
                     f"{pct(r['unknown_safe_fallback_rate'])} | {r['unknown_to_financial']} |")
    lines += ["", "*Accuracy among accepted* uses only the accepted commands as denominator; *overall correct routing* "
              "uses all legitimate test commands and counts rejections as failures. UNKNOWN_INTENT (rejected) and "
              "HELP_SUPPORT are counted separately; both are safe fallbacks.", "",
              "## Secondary analyses (not part of the primary comparison)", "",
              f"- Typo robustness: accuracy with one keyboard slip per command {pct(robust['accuracy'])} "
              f"(clean {pct(robust['clean_accuracy'])}), macro F1 {num(robust['macro_f1'])}."]
    tc = secondary["temperature_calibration"]
    lines.append(f"- Temperature scaling (fit on validation): T = {tc['temperature']:.3f}; test ECE "
                 f"{num(tc['test']['ece_raw'])} -> {num(tc['test']['ece_calibrated'])}, test log loss "
                 f"{num(tc['test']['log_loss_raw'])} -> {num(tc['test']['log_loss_calibrated'])}.")
    tt = secondary["three_tier_routing"]
    lines.append(f"- Three-tier routing (open >= {tt['open_threshold']:.3f}, menu >= {tt['menu_threshold']:.3f}): "
                 f"{pct(tt['known']['open_correct'])} of legitimate commands open the right workflow, "
                 f"{pct(tt['known']['menu_contains_intent'])} get a top-{tt['menu_size']} menu containing the right "
                 f"intent, {pct(tt['known']['open_wrong'])} open a wrong workflow; "
                 f"{pct(tt['unknown']['opened'])} of unknown commands open a workflow "
                 f"({pct(tt['unknown']['opened_financial'])} a money flow).")
    lines += ["", "## Figures", "", "![Confusion matrix](confusion_matrix.png)", "", "![ROC curves](roc_curves.png)", "",
              "![Threshold trade-off](threshold_sweep.png)", "",
              "## Provenance", "",
              f"- Training command: `{meta['training_command']}`; evaluation command: `{meta['evaluation_command']}`",
              f"- Trained on: {', '.join(meta['trained_on'])}; selected on: {', '.join(meta['selected_on'])}",
              f"- Artifact: `{meta['artifact']['directory']}` ({len(meta['artifact']['files'])} files, hashes in "
              "`experiment_metadata.json`)",
              f"- Git commit at evaluation: `{meta['git_commit']}`", ""]
    path = directory / "evaluation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
