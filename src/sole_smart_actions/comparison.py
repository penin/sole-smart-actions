"""Primary four-model comparison, built only from the generated ``results/<model>/`` files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import MODEL_NAMES
from .config import load_config, rel, resolve
from .evaluation import plots
from .evaluation.report import num, pct
from .utils import load_json, save_json, utc_now


def _row(name: str, d: Path) -> dict | None:
    meta_path = d / "experiment_metadata.json"
    if not meta_path.exists():
        return None
    meta, h = load_json(meta_path), load_json(meta_path)["headline"]
    sweep = pd.read_csv(d / "threshold_sweep.csv").set_index("threshold_label")
    lat = load_json(d / "latency_metrics.json") if (d / "latency_metrics.json").exists() else {}
    typo = load_json(d / "robustness_typos.json")
    dflt, none = sweep.loc["default"], sweep.loc["none"]
    return {
        "model": name, "display_name": meta["display_name"], "original_author": meta["original_author"],
        "primary_comparison": meta["primary_comparison"], "smoke_test": meta["smoke_test"],
        "trained_on": "+".join(meta["trained_on"]),
        "train_accuracy": h["train_accuracy"], "validation_accuracy": h["validation_accuracy"],
        "raw_test_accuracy": h["raw_test_accuracy"], "raw_test_correct": h["raw_test_correct"],
        "raw_test_n": h["raw_test_n"],
        "test_macro_precision": h["test_macro_precision"], "test_macro_recall": h["test_macro_recall"],
        "test_macro_f1": h["test_macro_f1"], "test_weighted_f1": h["test_weighted_f1"],
        "test_macro_roc_auc_ovr": h["test_macro_roc_auc_ovr"],
        "default_threshold": h["default_threshold"],
        "validation_coverage_at_default": meta["threshold_calibration"]["actual_validation_coverage"],
        "known_coverage_at_default": dflt["known_coverage"],
        "accuracy_among_accepted_at_default": dflt["accuracy_among_accepted"],
        "overall_correct_routing_at_default": dflt["overall_correct_routing_rate"],
        "unknown_rejection_at_default": dflt["unknown_rejection_rate"],
        "unknown_help_support_at_default": dflt["unknown_as_help_support_rate"],
        "unknown_safe_fallback_at_default": dflt["unknown_safe_fallback_rate"],
        "unknown_to_financial_at_default": int(dflt["unknown_to_financial"]),
        "unknown_to_financial_no_threshold": int(none["unknown_to_financial"]),
        "typo_test_accuracy": typo["accuracy"],
        "latency_p50_ms": lat.get("p50_ms"), "latency_p95_ms": lat.get("p95_ms"), "latency_device": lat.get("device"),
        "model_size_mb": meta["artifact"]["size_mb"], "n_parameters": meta["n_parameters"],
        "training_seconds_final_fit": meta["training_seconds_final_fit"],
        "training_seconds_incl_search": meta["training_wall_seconds_incl_search"],
        "hyperparameters": str(meta["hyperparameters"]),
        "dataset_sha256": meta["dataset"]["sha256"],
    }


def compare(results_root: str | Path | None = None, out: str | Path | None = None,
            models=MODEL_NAMES) -> pd.DataFrame:
    root = resolve(results_root or load_config()["paths"]["results_dir"])
    out = Path(out) if out else root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for m in models if (r := _row(m, root / m))]
    if not rows:
        raise FileNotFoundError(f"No evaluated models in {rel(root)}; run python -m scripts.evaluate --model all")
    df = pd.DataFrame(rows)
    if df["dataset_sha256"].nunique() != 1:
        raise RuntimeError("Models were evaluated on different datasets; refusing to compare")
    if df["primary_comparison"].eq(False).any() and not df["smoke_test"].all():
        raise RuntimeError("A non-primary run is in the primary results folder; refusing to mix protocols")
    missing = [m for m in models if m not in set(df["model"])]
    df.to_csv(out / "model_comparison.csv", index=False)
    save_json({"generated_utc": utc_now(), "models": rows, "missing": missing}, out / "model_comparison.json")

    plots.model_bars_png(df, "raw_test_accuracy", "Raw test accuracy (no threshold, n = 270)",
                         "Accuracy", out / "accuracy_comparison.png", ylim=(0, 1.08))
    plots.grouped_bars_png(df, [("test_macro_precision", "Macro precision"), ("test_macro_recall", "Macro recall"),
                                ("test_macro_f1", "Macro F1"), ("test_weighted_f1", "Weighted F1")],
                           "Test-set F1 and its components", "Score", out / "f1_comparison.png", fmt="{:.3f}")
    plots.model_bars_png(df, "latency_p50_ms", "Median single-command latency (P50, log scale)", "Milliseconds",
                         out / "latency_comparison.png", fmt="{:.2f} ms", log=True)
    plots.grouped_bars_png(df, [("unknown_rejection_at_default", "Rejected\n(UNKNOWN_INTENT)"),
                                ("unknown_help_support_at_default", "Accepted as\nHELP_SUPPORT"),
                                ("unknown_safe_fallback_at_default", "Safe fallback\n(either)")],
                           "300 unsupported commands at each model's validation default threshold", "Share",
                           out / "unknown_handling_comparison.png")
    _write_markdown(df, missing, out / "model_comparison.md")
    return df


def _write_markdown(df: pd.DataFrame, missing: list[str], path: Path) -> None:
    smoke = df["smoke_test"].any()
    L = ["# Sole Smart Actions - four-model comparison (primary protocol)", ""]
    if smoke:
        L += ["> **SMOKE TEST RESULTS** - tiny grids/subsets; not published experiments.", ""]
    L += [f"_Generated {utc_now()} from `results/<model>/` (actual predictions). All models: Dataset V5, "
          "trained on the 2,160 training commands only, selected and threshold-calibrated on the 270 validation "
          "commands, scored once on the same 270 test and 300 unknown-evaluation commands, nine intents in the "
          "same order._", ""]
    if missing:
        L += [f"> Not yet evaluated: {', '.join(missing)}.", ""]
    L += ["## Classification (raw, no confidence threshold)", "",
          "| Model | Raw test accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | ROC-AUC (OvR) | Train acc | Val acc |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['display_name']} | **{pct(r['raw_test_accuracy'])}** ({r['raw_test_correct']}/{r['raw_test_n']}) | "
                 f"{num(r['test_macro_precision'])} | {num(r['test_macro_recall'])} | **{num(r['test_macro_f1'])}** | "
                 f"{num(r['test_weighted_f1'])} | {num(r['test_macro_roc_auc_ovr'])} | {pct(r['train_accuracy'])} | "
                 f"{pct(r['validation_accuracy'])} |")
    L += ["", "## Routing at each model's validation-calibrated default threshold (95% validation coverage)", "",
          "| Model | Threshold | Test coverage | Accuracy among accepted | Overall correct routing | Unknown rejected | "
          "Unknown -> HELP_SUPPORT | Unknown safe fallback | Unknown -> money flow |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['display_name']} | {r['default_threshold']:.3f} | {pct(r['known_coverage_at_default'])} | "
                 f"{pct(r['accuracy_among_accepted_at_default'])} | {pct(r['overall_correct_routing_at_default'])} | "
                 f"{pct(r['unknown_rejection_at_default'])} | {pct(r['unknown_help_support_at_default'])} | "
                 f"{pct(r['unknown_safe_fallback_at_default'])} | {r['unknown_to_financial_at_default']} / 300 |")
    L += ["", "Thresholds differ by model because their probability distributions differ; each is calibrated the same "
          "way. *Accuracy among accepted* is selective accuracy and is never the headline accuracy.", "",
          "## Cost and robustness", "",
          "| Model | Parameters | Size | Latency P50 / P95 | Training (final fit) | Typo-test accuracy | Selected configuration |",
          "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        params = f"{int(r['n_parameters']):,}" if pd.notna(r["n_parameters"]) else "n/a"
        L.append(f"| {r['display_name']} | {params} | {r['model_size_mb']:.2f} MB | {num(r['latency_p50_ms'], 2)} / "
                 f"{num(r['latency_p95_ms'], 2)} ms ({r['latency_device']}) | {r['training_seconds_final_fit']:.1f} s | "
                 f"{pct(r['typo_test_accuracy'])} | `{r['hyperparameters']}` |")
    L += ["", "Latency is single-command end-to-end inference on the evaluation machine (see `experiment_metadata.json`); "
          "neural models ran on the GPU when available, so latencies are not hardware-neutral.", "",
          "## Caveats", "",
          "- The dataset is synthetic. No exact or normalized duplicates cross splits, but many test commands share "
          "sentence templates with training commands, so test accuracy overstates performance on real customer text.",
          "- 270 test commands: one command is 0.37 percentage points; differences of a few commands are within noise.",
          "- Unknown-intent handling uses only a confidence threshold; none of the models was trained on unsupported "
          "requests.", "",
          "![Accuracy](accuracy_comparison.png)", "", "![F1](f1_comparison.png)", "",
          "![Latency](latency_comparison.png)", "", "![Unknown handling](unknown_handling_comparison.png)", ""]
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")
