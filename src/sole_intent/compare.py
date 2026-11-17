"""Side-by-side comparison of two dataset versions of the same model (e.g. V4 vs V5).

Reads the reference version's published outputs read-only; writes only to the new version's results folder.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .preprocessing import dedup_key, template_key
from .reports import md_table, pct
from .train import load_bundle, reorder_proba
from .utils import OutputPaths, load_json, resolve_path, save_json, utc_now

_MONTHS = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b")


def date_template_key(text) -> str:
    """Number-masked template with month names also masked ("on 5 november" == "on 17 july")."""
    return _MONTHS.sub("<month>", template_key(text))


def controlled_subsets(cfg: dict, ctx: dict, ref_model_path, ref_test: pd.DataFrame) -> dict:
    """Score BOTH models on (a) test commands present in both versions' test sets and (b) the new version's
    replacement commands. (a) is the only like-for-like comparison of the two models."""
    labels = ctx["labels"]
    test = ctx["preds"]["test"].copy()
    keys = test["text"].map(dedup_key)
    test["in_reference_test"] = keys.isin(set(ref_test["text"].map(dedup_key)))
    replaced = set()
    sheet = cfg.get("validation", {}).get("replacement_log_sheet")
    if sheet:
        try:
            log = pd.read_excel(resolve_path(cfg["data"]["path"]), sheet_name=sheet)
            replaced = set(log["Replacement text"].map(dedup_key))
        except (ValueError, KeyError):
            pass
    test["replacement_command"] = keys.isin(replaced)
    train_tmpl = set(ctx["preds"]["train"]["text"].map(date_template_key))
    test["date_template_in_train"] = test["text"].map(date_template_key).isin(train_tmpl)

    models = {"reference": load_bundle(ref_model_path), "current": ctx["paths"]["tuned"]}
    models["current"] = load_bundle(models["current"])
    for name, bundle in models.items():
        proba = reorder_proba(bundle["pipeline"], test["text"], labels)
        test[f"pred_{name}"] = np.asarray(labels, dtype=object)[proba.argmax(axis=1)]
        test[f"correct_{name}"] = test[f"pred_{name}"] == test["actual_intent"]
    subsets = {
        "shared_with_reference_test": test["in_reference_test"],
        "replacement_commands": test["replacement_command"],
        "replacement_matching_train_date_template": test["replacement_command"] & test["date_template_in_train"],
        "all_test": pd.Series(True, index=test.index),
    }
    rows = [{"subset": k, "n": int(m.sum()),
             "reference_model_accuracy": float(test.loc[m, "correct_reference"].mean()) if m.any() else None,
             "current_model_accuracy": float(test.loc[m, "correct_current"].mean()) if m.any() else None,
             "current_correct": int(test.loc[m, "correct_current"].sum())} for k, m in subsets.items()]
    example = None
    matches = test[test["replacement_command"] & test["date_template_in_train"]]
    if len(matches):
        train = ctx["preds"]["train"]
        train_keys = train["text"].map(date_template_key)
        first = matches.iloc[0]["text"]
        example = (first, train.loc[train_keys == date_template_key(first), "text"].iloc[0])
    return {"table": pd.DataFrame(rows), "rows": test, "example": example}


def _reference(cfg: dict) -> dict:
    c = cfg["comparison"]
    base = resolve_path(c["reference_results_dir"])
    m = {s: load_json(base / "metrics" / f"metrics_{s}.json") for s in ("train", "validation", "test")}
    rej = load_json(base / "metrics" / "rejection_metrics.json")["primary_frozen_threshold_test_plus_unknown"]
    rt = load_json(base / "metrics" / "runtime_metrics.json")["tuned"]
    boot = load_json(base / "metrics" / "bootstrap_ci_test.json")["models"]["tuned"]
    leak = load_json(base / "metrics" / "metrics_test_leakage_controlled.json")
    test_pred = pd.read_csv(base / "predictions" / "test_predictions.csv")
    sweep = pd.read_csv(resolve_path(c["reference_threshold_analysis"]))
    default_row = sweep[sweep["is_baseline_v1"]].iloc[0]
    cfgref = load_json(base / "metrics" / "experiment_config.json")
    return {"metrics": m, "rejection": rej, "runtime": rt, "boot": boot, "leak": leak, "test_pred": test_pred,
            "safe_fallback": float(default_row["safe_fallback_rate"]),
            "unknown_to_financial": int(default_row["unknown_to_financial"]),
            "audit": load_json(base / "metrics" / "dataset_audit.json"),
            "params": cfgref["selected_params"], "dataset": cfgref["dataset"],
            # Historical records keep the path used at the time; the config may point to where it lives now.
            "model_path": resolve_path(c.get("reference_model_path") or cfgref["model_artifacts"]["tuned"])}


def _current(ctx: dict) -> dict:
    T = ctx["variants"]["tuned"]
    default_row = ctx["threshold_sweep"]["table"].query("is_default").iloc[0]
    return {"metrics": T["metrics"], "rejection": ctx["rejection"]["primary_frozen_threshold_test_plus_unknown"],
            "runtime": ctx["runtime"]["tuned"], "boot": ctx["bootstrap"]["models"]["tuned"],
            "leak": T["metrics"]["test_leakage_controlled"], "test_pred": ctx["preds"]["test"],
            "safe_fallback": float(default_row["safe_fallback_rate"]),
            "unknown_to_financial": int(default_row["unknown_to_financial"]),
            "audit": ctx["audit"], "params": ctx["best_params"], "dataset": ctx["fingerprint"]}


def _pid(p: dict) -> str:
    ng = p["ngram_range"]
    return f"ngram({ng[0]},{ng[1]}), C={p['C']:g}, {'binary' if p['binary'] else 'counts'}"


def write_version_comparison(cfg: dict, P: OutputPaths, ctx: dict) -> dict:
    c = cfg["comparison"]
    A, B = c["reference_label"], c["label"]
    ra, rb = _reference(cfg), _current(ctx)

    def row(name, a, b, fmt="pct", better="higher"):
        if fmt == "pct":
            fa, fb, d = pct(a), pct(b), f"{100 * (b - a):+.1f} pts"
        elif fmt == "ms":
            fa, fb, d = f"{a:.3f} ms", f"{b:.3f} ms", f"{b - a:+.3f} ms"
        elif fmt == "kb":
            fa, fb, d = f"{a * 1024:.1f} KB", f"{b * 1024:.1f} KB", f"{(b - a) * 1024:+.1f} KB"
        else:
            fa, fb, d = f"{a:.3f}", f"{b:.3f}", f"{b - a:+.3f}"
        return {"metric": name, A: fa, B: fb, "change": d}

    ma, mb = ra["metrics"], rb["metrics"]
    na, nb = len(ra["test_pred"]), len(rb["test_pred"])
    ca, cb = int(ra["test_pred"]["correct"].sum()), int(rb["test_pred"]["correct"].sum())
    rows = [
        row("Training accuracy", ma["train"]["accuracy"], mb["train"]["accuracy"]),
        row("Validation accuracy", ma["validation"]["accuracy"], mb["validation"]["accuracy"]),
        row("Raw test accuracy (no threshold)", ma["test"]["accuracy"], mb["test"]["accuracy"]),
        row("Test macro F1", ma["test"]["macro_f1"], mb["test"]["macro_f1"], "num"),
        row("Test weighted F1", ma["test"]["weighted_f1"], mb["test"]["weighted_f1"], "num"),
        row("Test macro ROC-AUC (OvR)", ma["test"]["macro_roc_auc_ovr"], mb["test"]["macro_roc_auc_ovr"], "num"),
        row("Validation-calibrated threshold", ra["rejection"]["threshold"], rb["rejection"]["threshold"]),
        row("Known-intent coverage at default (test)", ra["rejection"]["known_intent_coverage"],
            rb["rejection"]["known_intent_coverage"]),
        row("Unknown rejection rate at default", ra["rejection"]["unknown_rejection_rate"],
            rb["rejection"]["unknown_rejection_rate"]),
        row("Unknown safe-fallback rate at default", ra["safe_fallback"], rb["safe_fallback"]),
        row("Inference latency P50 (single request)", ra["runtime"]["latency_single_request"]["median_ms"],
            rb["runtime"]["latency_single_request"]["median_ms"], "ms"),
        row("Inference latency P95 (single request)", ra["runtime"]["latency_single_request"]["p95_ms"],
            rb["runtime"]["latency_single_request"]["p95_ms"], "ms"),
        row("Model size", ra["runtime"]["model_size_mb"], rb["runtime"]["model_size_mb"], "kb"),
    ]
    table = pd.DataFrame(rows)
    extra = pd.DataFrame([
        {"metric": "Correct / total test commands", A: f"{ca} / {na}", B: f"{cb} / {nb}", "change": f"{cb - ca:+d}"},
        {"metric": "Test macro F1, 95% phrase-clustered CI",
         A: f"{ra['boot']['macro_f1']['ci_low']:.3f}–{ra['boot']['macro_f1']['ci_high']:.3f}",
         B: f"{rb['boot']['macro_f1']['ci_low']:.3f}–{rb['boot']['macro_f1']['ci_high']:.3f}", "change": ""},
        {"metric": "Leakage-controlled test macro F1", A: f"{ra['leak']['macro_f1']:.3f} (n={ra['leak']['n_samples']})",
         B: f"{rb['leak']['macro_f1']:.3f} (n={rb['leak']['n_samples']})", "change": ""},
        {"metric": "Unknowns routed to money-moving intents at default", A: ra["unknown_to_financial"],
         B: rb["unknown_to_financial"], "change": f"{rb['unknown_to_financial'] - ra['unknown_to_financial']:+d}"},
        {"metric": "Selected hyperparameters", A: _pid(ra["params"]), B: _pid(rb["params"]), "change": ""},
        {"metric": "Vocabulary size", A: ra["runtime"]["vocabulary_size"], B: rb["runtime"]["vocabulary_size"],
         "change": f"{rb['runtime']['vocabulary_size'] - ra['runtime']['vocabulary_size']:+d}"},
        {"metric": "Training time (final fit)", A: f"{ra['runtime']['training_seconds'] * 1000:.0f} ms",
         B: f"{rb['runtime']['training_seconds'] * 1000:.0f} ms", "change": ""},
    ])

    ctl = controlled_subsets(cfg, ctx, ra["model_path"], ra["test_pred"])
    ctl["rows"].to_csv(P.metrics / "test_original_vs_replacement_commands.csv", index=False)
    ctl["table"].to_csv(P.metrics / "controlled_model_comparison.csv", index=False)
    ct = ctl["table"].set_index("subset")
    sh, rp = ct.loc["shared_with_reference_test"], ct.loc["replacement_commands"]
    rpt = ct.loc["replacement_matching_train_date_template"]
    ctl_view = pd.DataFrame([{
        "test subset": {"shared_with_reference_test": f"Commands in BOTH test sets (identical examples)",
                        "replacement_commands": f"{B} replacement commands (new in {B})",
                        "replacement_matching_train_date_template": "  …of which match a training command once numbers + months are masked",
                        "all_test": f"All {B} test commands"}[k],
        "n": int(r["n"]), f"{A} model": pct(r["reference_model_accuracy"]), f"{B} model": pct(r["current_model_accuracy"])}
        for k, r in ct.iterrows()])

    # How different are the test sets?
    ka = set(ra["test_pred"]["text"].map(dedup_key))
    kb = set(rb["test_pred"]["text"].map(dedup_key))
    shared_test = len(ka & kb)
    aa, ab = ra["audit"], rb["audit"]
    gap = {lab: {s: m[s]["accuracy"] for s in ("train", "validation", "test")} for lab, m in ((A, ma), (B, mb))}
    gap_a = gap[A]["train"] - gap[A]["test"]
    gap_b = gap[B]["train"] - gap[B]["test"]
    lc_b = ctx["learning_curve"].iloc[-1]
    delta_acc = mb["test"]["accuracy"] - ma["test"]["accuracy"]
    ci_overlap = not (rb["boot"]["macro_f1"]["ci_low"] > ra["boot"]["macro_f1"]["ci_high"]
                      or ra["boot"]["macro_f1"]["ci_low"] > rb["boot"]["macro_f1"]["ci_high"])

    L = [
        f"# {A} vs {B}: Bag of Words + Logistic Regression", "",
        f"_Generated {utc_now()}. {A}: `{ra['dataset']['dataset_version']}` (outputs in `{c['reference_results_dir']}/`, "
        f"read-only). {B}: `{rb['dataset']['dataset_version']}` (outputs in `{P.rel(P.results)}/`). Same code, "
        "hyperparameter grid, preprocessing, seed (42) and selection rules; only the dataset differs._", "",
        "## Headline comparison", "",
        "Raw test accuracy counts every one of the legitimate test commands, with no confidence threshold applied. "
        "Threshold rows use each version's own validation-calibrated default (95% validation coverage target).", "",
        md_table(table),
        "Additional detail:", "",
        md_table(extra),
        "## Controlled comparison: same test commands, both models", "",
        f"Because the test sets differ, both trained models were also scored on identical commands taken from the "
        f"{B} test set. The {A} model was loaded read-only from `{P.rel(ra['model_path'])}`.", "",
        md_table(ctl_view),
        f"- **Like-for-like improvement:** on the {int(sh['n'])} commands present in both test sets, accuracy goes from "
        f"{pct(sh['reference_model_accuracy'])} ({A} model) to {pct(sh['current_model_accuracy'])} ({B} model), "
        f"{100 * (sh['current_model_accuracy'] - sh['reference_model_accuracy']):+.1f} points. This is the change "
        "attributable to the new training data rather than to a different test set.",
        f"- **The replacement commands are easier:** the {B} model scores {pct(rp['current_model_accuracy'])} on the "
        f"{int(rp['n'])} replacement test commands. {int(rpt['n'])} of them match a training command once numbers and "
        "month names are masked"
        + (f" (e.g. test \"{ctl['example'][0]}\" vs training \"{ctl['example'][1]}\")" if ctl["example"] else "")
        + ". The deduplication "
        "removed exact repeats but introduced template-generated commands shared across splits. That is a milder "
        "form of the same leakage, and it inflates the headline test accuracy.",
        f"- **Headline vs like-for-like:** raw test accuracy rose {100 * (mb['test']['accuracy'] - ma['test']['accuracy']):.1f} "
        f"points, but only {100 * (sh['current_model_accuracy'] - sh['reference_model_accuracy']):.1f} of that is visible on "
        f"identical examples. Report {B} accuracy on the {int(sh['n'])} original commands "
        f"({pct(sh['current_model_accuracy'])}) alongside the headline figure.",
        f"- Per-command detail: `{P.rel(P.metrics / 'test_original_vs_replacement_commands.csv')}`.", "",
        "## Read this before comparing the numbers", "",
        f"**{A} and {B} are evaluated on different test examples.** Only {shared_test} of the {nb} {B} test commands "
        f"also appear in the {A} test set (after normalization); the rest were replaced when duplicates were "
        f"removed. A change in test accuracy therefore mixes three effects: (1) the model learning from different "
        f"training data, (2) a different, possibly harder or easier, test set, and (3) the removal of "
        f"train→test duplicates that inflated {A}. It cannot be attributed to model improvement alone. With "
        f"{nb} test commands, one command is worth {100 / nb:.2f} accuracy points. See the controlled comparison above.", "",
        f"- {A} test set: {aa['unique_phrases_per_split']['test']} distinct phrases among {na} rows; "
        f"{aa['leakage']['test']['leakage_flagged']} rows repeated a training phrase (exact or number-masked).",
        f"- {B} test set: {ab['unique_phrases_per_split']['test']} distinct phrases among {nb} rows; "
        f"{ab['leakage']['test']['normalized_in_train']} rows repeat a training phrase exactly, "
        f"{ab['leakage']['test']['template_in_train']} once numbers are masked.",
        f"- The macro F1 95% confidence intervals {'overlap' if ci_overlap else 'do not overlap'} "
        f"({A} {ra['boot']['macro_f1']['ci_low']:.3f}–{ra['boot']['macro_f1']['ci_high']:.3f}, "
        f"{B} {rb['boot']['macro_f1']['ci_low']:.3f}–{rb['boot']['macro_f1']['ci_high']:.3f})"
        + (", so the difference is within sampling noise." if ci_overlap else "."), "",
        "## Is there overfitting?", "",
        f"| | Train acc. | Validation acc. | Test acc. | Train − test |", "|---|---|---|---|---|",
        *[f"| {lab} | {pct(g['train'])} | {pct(g['validation'])} | {pct(g['test'])} | "
          f"{100 * (g['train'] - g['test']):.1f} pts |" for lab, g in gap.items()], "",
        f"The train–test gap {'narrowed' if gap_b < gap_a else 'widened'} from {100 * gap_a:.1f} to {100 * gap_b:.1f} "
        f"points. Training accuracy near 100% is expected for a linear model on 2,160 short commands with "
        f"{rb['runtime']['vocabulary_size']:,} features ({A}: {ra['runtime']['vocabulary_size']:,}), so the gap measures how well the training vocabulary transfers to new phrasings "
        f"more than classic memorization. Inside the {B} training split, grouped 5-fold cross-validation reaches "
        f"{lc_b['cv_mean']:.3f} macro F1 at {int(lc_b['train_size'])} examples, versus "
        f"{mb['validation']['macro_f1']:.3f} on validation and {mb['test']['macro_f1']:.3f} on test. "
        + ("The held-out splits are clearly harder than the training distribution, so the remaining gap reflects "
           "distribution shift between splits (new paraphrase templates) as well as fitting to training phrases."
           if lc_b['cv_mean'] - mb['test']['macro_f1'] > 0.05 else
           "Cross-validated and held-out performance are close, which suggests the splits are similar in difficulty."),
        "",
        "## Did duplicate removal affect generalization?", "",
        f"- {A} contained {aa['normalized_duplicate_redundant_rows']} redundant rows under a punctuation-insensitive "
        f"key; {B} contains {ab['normalized_duplicate_redundant_rows']} (pairs differing only by interior punctuation, "
        "e.g. \"App keeps crashing? asap\" / \"App keeps crashing asap\"; 0 under the required "
        "surrounding-punctuation definition). Removing duplicates means each training "
        "example is a distinct phrasing, and test scores can no longer be inflated by memorized repeats.",
        f"- Raw test accuracy {'rose' if delta_acc > 0 else 'fell' if delta_acc < 0 else 'did not change'} by "
        f"{100 * abs(delta_acc):.1f} points ({ca}/{na} → {cb}/{nb}). Because the test sets differ (see above), this "
        "is evidence about the *combined* effect of the new data, not a controlled measurement of deduplication.",
        f"- The leakage-controlled test subsets (rows not matching any training phrase) give macro F1 "
        f"{ra['leak']['macro_f1']:.3f} ({A}) vs {rb['leak']['macro_f1']:.3f} ({B}); this is the fairest single "
        "comparison of generalization to unseen commands.", "",
        "## Caveats", "",
        "- Both datasets are synthetic; neither test set measures real Fijian customer language.",
        f"- Each version's threshold was calibrated on its own validation split. The unknown-command set is "
        f"{'identical' if aa['unique_phrases_per_split']['evaluation_only'] == ab['unique_phrases_per_split']['evaluation_only'] else 'different'} "
        f"in distinct-phrase count ({aa['unique_phrases_per_split']['evaluation_only']} vs "
        f"{ab['unique_phrases_per_split']['evaluation_only']}), so unknown-rejection rates are also not strictly comparable.",
        "- Latency is a single-machine wall-clock measurement and varies run to run.", "",
    ]
    out = P.results / f"{A.lower()}_vs_{B.lower()}_comparison.md"
    out.write_text("\n".join(L), encoding="utf-8")
    save_json({"generated_utc": utc_now(), "table": rows, "extra": extra.to_dict(orient="records"),
               "shared_test_commands": shared_test, "train_test_gap": {A: gap_a, B: gap_b},
               "ci_overlap": ci_overlap, "controlled_subsets": ctl["table"].to_dict(orient="records")}, P.results / f"{A.lower()}_vs_{B.lower()}_comparison.json")
    return {"path": out, "table": table}
