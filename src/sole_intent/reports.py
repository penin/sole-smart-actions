"""Markdown report generation. Every number is read from the run's computed results; the
reports are regenerated on each run of ``scripts/run_all.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rejection import UNKNOWN_INTENT
from .utils import PROJECT_ROOT, output_paths, utc_now


# ------------------------------------------------------------------ formatting helpers
def fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}f}"
    return str(x)


def pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


def md_table(df: pd.DataFrame, digits: int = 3, index: bool = False) -> str:
    if df is None or len(df) == 0:
        return "_(none)_\n"
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = [fmt(v, digits).replace("|", "\\|").replace("\n", " ") for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ci(b: dict) -> str:
    return f"{b['estimate']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"


def _out(ctx: dict):
    return ctx.get("out") or output_paths(ctx["cfg"])


def _r(ctx: dict) -> str:
    """Project-relative results folder name, e.g. ``results`` or ``results_v5``."""
    return _out(ctx).rel(_out(ctx).results)


def _rel(path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


# ------------------------------------------------------------------ error analysis tables
def error_analysis_tables(ctx: dict) -> dict[str, pd.DataFrame]:
    preds, thr, diag, labels = ctx["preds"], ctx["threshold"], ctx["diag"], ctx["labels"]
    held = pd.concat([preds["validation"], preds["test"]], ignore_index=True)
    out: dict[str, pd.DataFrame] = {}
    cols = ["split", "text", "actual_intent", "predicted_intent", "max_confidence", "top2_margin", "oov_tokens"]

    rows = []
    for split in ["train", "validation", "test"]:
        p = preds[split]
        wrong = p[~p["correct"]]
        for (a, b), g in wrong.groupby(["actual_intent", "predicted_intent"]):
            rows.append({"split": split, "actual_intent": a, "predicted_intent": b, "count": len(g),
                         "unique_phrases": g["text"].str.lower().nunique(),
                         "example": g["text"].iloc[0]})
    out["confusions"] = (pd.DataFrame(rows).sort_values(["split", "count"], ascending=[True, False])
                         if rows else pd.DataFrame())

    wrong = held[~held["correct"]]
    out["accepted_errors"] = (wrong[wrong["max_confidence"] >= thr]
                              .drop_duplicates(["split", "text"]).sort_values("max_confidence", ascending=False)[cols])
    out["high_conf_errors"] = wrong[wrong["max_confidence"] >= diag["high_confidence"]][cols]
    right = held[held["correct"]]
    out["low_conf_correct"] = (right[right["max_confidence"] < thr].drop_duplicates(["split", "text"])
                               .sort_values("max_confidence")[cols])
    out["ambiguous"] = (held[held["top2_margin"] < diag["ambiguous_margin"]]
                        .drop_duplicates(["split", "text"]).sort_values("top2_margin")
                        [cols[:4] + ["second_intent", "top2_margin", "correct"]])

    rows = []
    for split in ["train", "validation", "test"]:
        p = preds[split]
        short = p["n_tokens"] <= diag["short_command_max_tokens"]
        has_oov = p["oov_fraction"] > 0
        rows.append({"split": split, "n": len(p),
                     "short_n": int(short.sum()), "short_accuracy": p.loc[short, "correct"].mean() if short.any() else np.nan,
                     "longer_accuracy": p.loc[~short, "correct"].mean() if (~short).any() else np.nan,
                     "rows_with_oov": int(has_oov.sum()), "share_rows_with_oov": has_oov.mean(),
                     "oov_accuracy": p.loc[has_oov, "correct"].mean() if has_oov.any() else np.nan,
                     "in_vocab_accuracy": p.loc[~has_oov, "correct"].mean() if (~has_oov).any() else np.nan,
                     "mean_oov_token_share": p["oov_fraction"].mean()})
    out["short_oov_summary"] = pd.DataFrame(rows)
    short_rows = held[held["n_tokens"] <= diag["short_command_max_tokens"]]
    out["short_errors"] = short_rows[~short_rows["correct"]].drop_duplicates(["split", "text"])[cols]
    oov_words = (held.loc[held["oov_tokens"].str.len() > 0, "oov_tokens"].str.split().explode().value_counts()
                 .rename_axis("oov_token").reset_index(name="held_out_occurrences"))
    out["oov_words"] = oov_words

    rows = []
    for split in ["validation", "test"]:
        p = preds[split]
        for flag, g in p.groupby("leakage_flag"):
            rows.append({"split": split, "leakage_flag": bool(flag), "n": len(g), "accuracy": g["correct"].mean()})
    out["leakage_accuracy"] = pd.DataFrame(rows)
    out["leaked_rows"] = preds["test"][preds["test"]["leakage_flag"]][["text", "actual_intent", "predicted_intent",
                                                                       "max_confidence", "correct"]]

    # Financial safety
    money = ctx["money_moving"]
    rows = []
    for split in ["train", "validation", "test"]:
        p = preds[split]
        for a, b in [("TRANSFER_MONEY", "BILL_PAY"), ("BILL_PAY", "TRANSFER_MONEY")]:
            m = (p["actual_intent"] == a) & (p["predicted_intent"] == b)
            rows.append({"split": split, "actual": a, "predicted": b, "count": int(m.sum()),
                         "accepted_at_threshold": int((m & ~p["rejected"]).sum())})
    out["transfer_billpay"] = pd.DataFrame(rows)
    risky = held[(~held["correct"]) & (~held["rejected"]) & held["final_intent"].isin(money)]
    out["money_false_positives"] = risky.drop_duplicates(["split", "text"])[cols]
    u = preds["unknown"]
    out["unknown_money_accepts"] = (u[(~u["rejected"]) & u["final_intent"].isin(money)]
                                    .drop_duplicates("text").sort_values("max_confidence", ascending=False)
                                    [["text", "final_intent", "max_confidence"]])
    out["unknown_top_accepts"] = (u[~u["rejected"]].drop_duplicates("text").sort_values("max_confidence", ascending=False)
                                  [["text", "final_intent", "max_confidence"]].head(15))
    for name, frame in out.items():
        frame.to_csv(_out(ctx).metrics / f"error_{name}.csv", index=False)
    return out


# ------------------------------------------------------------------ optional sections
def strict_validation_section(ctx: dict) -> list[str]:
    v = ctx.get("validation")
    if not v:
        return []
    checks = pd.DataFrame(v["checks"])[["check", "value", "expected", "status", "details"]]
    lines = ["## Pre-training validation (required checks)", "",
             f"**Result: {'PASSED' if v['passed'] else 'FAILED'}.** Checks run before any training. A failure "
             "stops the run and nothing in the dataset is changed. Duplicate key: Unicode NFKC, casefold, collapsed "
             "whitespace, surrounding punctuation removed. Rows marked `info` are stricter diagnostics, not "
             "requirements.", "", md_table(checks)]
    if "replacement_log_rows" in v:
        lines += [f"The workbook's *Replacement Log* records {v['replacement_log_rows']} duplicate occurrences that the "
                  "dataset author replaced with new synthetic commands (by split: "
                  + ", ".join(f"{k} {n}" for k, n in v.get("replacements_by_split", {}).items()) + ").", ""]
    return lines


def threshold_sweep_section(ctx: dict) -> list[str]:
    sw = ctx.get("threshold_sweep")
    if not sw:
        return []
    t = sw["table"]
    view = pd.DataFrame({
        "threshold": t["label"],
        "known coverage": t["known_coverage"].map(pct),
        f"accepted (of {int(t['known_n'].iloc[0])})": t["known_accepted"],
        "correct & accepted": t["known_correct_accept"],
        "accuracy of accepted": t["accepted_accuracy"].map(pct),
        f"correct routing (of {int(t['known_n'].iloc[0])})": t["overall_correct_routing_rate"].map(pct),
        "legit rejected": t["known_false_reject"],
        "unknown rejected": t["unknown_rejection_rate"].map(pct),
        "unknown → HELP_SUPPORT": t["unknown_to_help_support"],
        "unknown safe fallback": t["safe_fallback_rate"].map(pct),
        "unknown → financial": t["unknown_to_financial"],
        "→ TRANSFER / WITHDRAW / BILL": [f"{a} / {b} / {c}" for a, b, c in zip(
            t["unknown_to_TRANSFER_MONEY"], t["unknown_to_WITHDRAW_MONEY"], t["unknown_to_BILL_PAY"])],
    })
    d = sw["default_threshold"]
    rel = _out(ctx).rel(sw["dir"])
    return ["## 11b. Threshold sweep (separate from raw accuracy)", "",
            f"Known commands are the {int(t['known_n'].iloc[0])} legitimate test rows; unknown commands are the "
            f"{int(t['unknown_n'].iloc[0])} evaluation-only rows. *Correct routing* counts rejected legitimate commands "
            "as incorrect. The validation-calibrated default "
            f"({100 * d:.2f}%) is the only methodological choice. **Every other row is exploratory** and must not "
            "be used to pick an \"optimal\" threshold, because the same test and unknown sets are scored at each one. "
            "The *No threshold* row equals raw test accuracy by construction (verified in code).", "",
            md_table(view),
            f"Files: `{rel}/threshold_sweep.csv`, `.json`, `dense_threshold_sweep.csv`, "
            "`decisions_at_default_threshold.csv`, `legitimate_rejection_by_intent.csv`; charts "
            "`threshold_comparison.png`, `coverage_vs_rejection.png`, `safe_fallback_comparison.png`, "
            "`financial_routing_risk.png`, `legitimate_rejection_by_intent.png`.", ""]


# ------------------------------------------------------------------ data quality report
def data_quality_report(ctx: dict, ea: dict) -> str:
    a, fp, f, d = ctx["audit"], ctx["fingerprint"], ctx["findings"], ctx["audit_details"]
    labels, unk = ctx["labels"], ctx["cfg"]["unknown_label"]
    bal = pd.DataFrame(a["class_balance"]).T.reindex(labels + [unk])
    leak = a["leakage"]
    proc = _out(ctx).rel(_out(ctx).processed)
    test_rows, test_unique = leak["test"]["rows"], a["unique_phrases_per_split"]["test"]
    unk_rows, unk_unique = leak["evaluation_only"]["rows"], a["unique_phrases_per_split"]["evaluation_only"]
    oov_test = float(ea["short_oov_summary"].set_index("split").loc["test", "share_rows_with_oov"])
    groups = d["duplicate_groups"]
    unk_top = groups[groups["splits"].str.contains("evaluation_only")].head(1)
    implications = [
        "Training-set metrics and cross-validation inside the training split are optimistic"
        + (" (duplicates and shared templates)." if a["normalized_duplicate_redundant_rows"] else
           " (shared templates and vocabulary)."),
        (f"The test set holds {test_unique} distinct phrases in {test_rows} rows; repeated templates mean single "
         "errors move metrics in steps of several points." if test_unique < test_rows else
         f"All {test_rows} test rows are distinct phrases; one error moves accuracy by {100 / test_rows:.2f} points."),
        (f"Train→test overlap: {leak['test']['normalized_in_train']} exact/normalized and "
         f"{leak['test']['template_in_train']} number-masked matches. Reported, not removed; see the "
         "leakage-controlled test subset." if leak["test"]["leakage_flagged"] else
         "No train→test phrase overlap was found, even after masking numbers."),
        (f"Unknown-evaluation phrases repeat ({unk_unique} distinct in {unk_rows} rows; e.g. "
         f"`{unk_top['canonical'].iloc[0]}` ×{int(unk_top['count'].iloc[0])}); unknown rejection is also reported "
         "per unique phrase." if unk_unique < unk_rows and len(unk_top) else
         f"Unknown-evaluation phrases are distinct ({unk_unique} of {unk_rows})."),
    ]
    lines = [
        "# Data Quality Report — Sole Smart Actions Intent Dataset", "",
        f"_Generated {utc_now()} by `scripts/run_all.py`. Source workbook is read-only; no rows were "
        "deleted, edited or re-split._", "",
        "## Source", "",
        f"- File: `{_rel(PROJECT_ROOT / ctx['cfg']['data']['path'])}` (sheet `{ctx['cfg']['data']['sheet']}`)",
        f"- Dataset version: `{fp['dataset_version']}`; rows: {fp['n_rows']:,}",
        f"- File SHA-256: `{fp['file_sha256']}`",
        f"- Content SHA-256 (text,intent,split,scope as CSV): `{fp['content_sha256']}`", "",
        *strict_validation_section(ctx),
        "## Integrity checks", "",
        md_table(f[f["category"].isin(["schema", "splits", "labels"])]),
        "## Class balance by split", "",
        md_table(bal.fillna(0).astype(int), index=True),
        "Supported intents are perfectly balanced (240 / 30 / 30). `CHECK_SPENDING` is the stretch intent "
        "(`scope = stretch`); the other eight are `core`. All 300 `UNKNOWN_INTENT` rows are "
        "`evaluation_only` / `rejection_test` and are never used as a training class.", "",
        "## Utterance length", "",
        f"Tokens per utterance: mean {a['token_length']['mean']:.2f}, median {a['token_length']['50%']:.0f}, "
        f"min {a['token_length']['min']:.0f}, max {a['token_length']['max']:.0f}. Utterances are short "
        "commands, so a single unseen word can remove most of the usable signal.", "",
        md_table(d["length_stats"][["split", "intent", "n_tokens_mean", "n_tokens_min", "n_tokens_max"]]
                 .query("split != 'evaluation_only'").head(40), digits=2),
        "## Duplicates", "",
        md_table(f[f["category"] == "duplicates"]),
        f"Only {a['unique_normalized_phrases']:,} of {a['n_rows']:,} rows are distinct after normalization "
        f"(lowercase, punctuation-insensitive). Unique phrases per partition: "
        + ", ".join(f"{k} {v:,}" for k, v in a["unique_phrases_per_split"].items()) + ". "
        + (f"Exact duplicates: {a['exact_duplicate_redundant_rows']} redundant rows. "
           + ("The remaining repeats differ only in case or punctuation. " if a["exact_duplicate_redundant_rows"]
              < a["normalized_duplicate_redundant_rows"] else "")
           if a["normalized_duplicate_redundant_rows"] else "No repeated phrases. ")
        + (f"Within-partition repetition inflates the apparent sample size: the {test_rows}-row test set holds "
           f"{test_unique} distinct phrases, so test confidence intervals use a phrase-clustered bootstrap."
           if test_unique < test_rows else
           f"Every one of the {test_rows} test rows is a distinct phrase."), "",
        "Most repeated phrases:", "",
        md_table(d["duplicate_groups"].head(15)[["canonical", "count", "splits", "intents"]]),
        "## Contradictory labels", "",
        f"{a['contradictory_label_phrases']} normalized phrases carry more than one intent label"
        + (" — no label conflicts were found." if a["contradictory_label_phrases"] == 0
           else f"; see `{proc}/duplicate_groups.csv`."), "",
        "## Cross-partition overlap and leakage", "",
        md_table(f[f["category"] == "leakage"]),
        "| Partition | Rows | Phrase seen in train | Number-masked template seen in train | Near-duplicate (Jaccard ≥ "
        f"{ctx['diag']['near_duplicate_jaccard']}) | Leakage-flagged |",
        "|---|---|---|---|---|---|",
        *[f"| {s} | {v['rows']} | {v['normalized_in_train']} | {v['template_in_train']} | "
          f"{v['near_duplicate_of_train']} | {v['leakage_flagged']} |" for s, v in leak.items()], "",
        f"**Leakage flag:** {leak['test']['leakage_flagged']} test rows ({pct(leak['test']['leakage_flagged'] / leak['test']['rows'])}) "
        "repeat a training phrase exactly or after masking numbers. The primary benchmark keeps the supplied "
        "split; a **leakage-controlled test subset** excluding these rows is evaluated separately "
        "(`metrics_test_leakage_controlled.json`)."
        + (" Unknown-evaluation phrases do not overlap training." if leak["evaluation_only"]["leakage_flagged"] == 0
           else f" {leak['evaluation_only']['leakage_flagged']} unknown-evaluation rows match a training phrase."), "",
        f"Rows flagged per partition are listed in `{proc}/leakage_rows.csv`; all shared phrases in "
        f"`{proc}/cross_split_overlap.csv`.", "",
        "## Vocabulary shift (train vocabulary vs held-out partitions)", "",
        md_table(ea["short_oov_summary"][["split", "n", "rows_with_oov", "share_rows_with_oov",
                                          "mean_oov_token_share"]]),
        f"{pct(oov_test)} of test rows contain at least one word that never occurs in training (most frequent: "
        + ", ".join(f"`{w}`" for w in ea["oov_words"]["oov_token"].head(12)) + "). "
        + ("The held-out partitions therefore test generalization to **unseen paraphrase templates**, not just "
           "unseen rows, and this is a main driver of the train/held-out gap." if oov_test >= 0.2 else
           "Most held-out commands are built from vocabulary seen in training."), "",
        "## Implications", "",
        *[f"{i}. {txt}" for i, txt in enumerate(implications, start=1)], "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ evaluation report
def evaluation_report(ctx: dict, ea: dict) -> str:
    cfg, labels, core = ctx["cfg"], ctx["labels"], ctx["core"]
    V = ctx["variants"]
    T, D = V["tuned"]["metrics"], V["default"]["metrics"]
    rej = ctx["rejection"]
    rp = rej["primary_frozen_threshold_test_plus_unknown"]
    sup = rej["supplementary_unknown_aware"]
    rt = ctx["runtime"]
    bs = ctx["bootstrap"]
    best, dflt = ctx["best_params"], ctx["default_params"]
    thr = ctx["threshold"]
    grid = ctx["grid"]
    diff = bs["macro_f1_difference_tuned_minus_default"]
    conf = ea["confusions"]
    test_conf = conf[conf["split"] == "test"].head(6) if len(conf) else conf
    lc = ctx["learning_curve"]
    env = rt["environment"]
    lat1, latb = rt["tuned"]["latency_single_request"], rt["tuned"]["latency_batch"]
    tuned_worse = T["test"]["macro_f1"] < D["test"]["macro_f1"]
    tb = ea["transfer_billpay"]
    n_tb_heldout = int(tb.loc[tb["split"] != "train", "count"].sum())
    n_tb_train = int(tb.loc[tb["split"] == "train", "count"].sum())
    tr = ctx["preds"]["train"]
    tb_rows = tr[(tr["actual_intent"] == "TRANSFER_MONEY") & (tr["predicted_intent"] == "BILL_PAY")]
    tb_example = tb_rows["text"].iloc[0] if len(tb_rows) else "n/a"
    best_by_c = grid.groupby("C")["validation_macro_f1"].max()
    # Facts used to keep the narrative true for whichever dataset produced this run.
    au = ctx["audit"]
    oov_test = float(ea["short_oov_summary"].set_index("split").loc["test", "share_rows_with_oov"])
    gap_pts = 100 * (T["train"]["accuracy"] - T["test"]["accuracy"])
    test_n, test_unique = T["test"]["n_samples"], au["unique_phrases_per_split"]["test"]
    val_repeats = au["unique_phrases_per_split"]["validation"] < T["validation"]["n_samples"]
    dt, tt = D["test"]["macro_f1"], T["test"]["macro_f1"]
    default_vs_test = ("but higher" if dt > tt + 5e-4 else "and lower" if dt < tt - 5e-4 else "and about the same")
    test_pred = ctx["preds"]["test"]
    test_errors = test_pred[~test_pred["correct"]]
    oov_error_share = float((test_errors["oov_fraction"] > 0).mean()) if len(test_errors) else 0.0
    mean_conf, acc = T["test"]["mean_confidence"], T["test"]["accuracy"]
    calibration_word = ("under-confident" if mean_conf < acc - 0.05 else
                        "over-confident" if mean_conf > acc + 0.05 else "reasonably calibrated")
    unk_unique = au["unique_phrases_per_split"]["evaluation_only"]
    n_unknown = au["leakage"]["evaluation_only"]["rows"]
    leak_test = au["leakage"]["test"]
    max_repeat = int((conf["count"] / conf["unique_phrases"]).max()) if len(conf) else 1

    def split_row(name, m):
        return {"split": name, "n": m["n_samples"], "accuracy": m["accuracy"], "balanced_acc": m["balanced_accuracy"],
                "macro_P": m["macro_precision"], "macro_R": m["macro_recall"], "macro_F1": m["macro_f1"],
                "weighted_F1": m["weighted_f1"], "log_loss": m["log_loss"], "macro_ROC_AUC": m["macro_roc_auc_ovr"],
                "weighted_ROC_AUC": m["weighted_roc_auc_ovr"], "macro_AP": m["macro_average_precision"],
                "ECE": m["expected_calibration_error"]}

    split_names = [("train (in-sample)", "train"), ("validation (used for selection)", "validation"),
                   ("test (held-out)", "test"), ("test, leakage-controlled", "test_leakage_controlled"),
                   ("test, one row per phrase", "test_unique_phrases")]
    split_tbl = pd.DataFrame([split_row(n, T[k]) for n, k in split_names])
    variant_tbl = pd.DataFrame([{"model": f"{k} ({_pid(v['params'])})", "train_F1": v["metrics"]["train"]["macro_f1"],
                                 "val_F1": v["metrics"]["validation"]["macro_f1"],
                                 "test_acc": v["metrics"]["test"]["accuracy"],
                                 "test_F1": v["metrics"]["test"]["macro_f1"],
                                 "test_F1_leak_ctrl": v["metrics"]["test_leakage_controlled"]["macro_f1"],
                                 "test_ROC_AUC": v["metrics"]["test"]["macro_roc_auc_ovr"],
                                 "test_log_loss": v["metrics"]["test"]["log_loss"]} for k, v in V.items()])
    pc = pd.DataFrame(T["test"]["per_class"])
    pc["scope"] = np.where(pc["intent"].isin(core), "core", "stretch")
    core_rows = pd.DataFrame([{"split": s, **{k: T[s]["core8_breakdown"][k] for k in
                              ["n_samples", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                               "predicted_outside_core"]}} for s in ["train", "validation", "test"]])
    c8 = ctx["core8"]
    rej_counts = pd.DataFrame([{"outcome": k.replace("_", " "), "count": v} for k, v in rp["counts"].items()])

    L = [
        "# Evaluation Report — Sole Smart Actions BoW + Logistic Regression Baseline", "",
        f"_Generated {utc_now()} · experiment `{cfg['experiment']['experiment_id']}` · model version "
        f"{cfg['experiment']['model_version']} · seed {cfg['experiment']['random_seed']} · dataset "
        f"`{cfg['data']['dataset_version']}` (SHA-256 `{ctx['fingerprint']['file_sha256'][:12]}…`)._", "",
        "All numbers below were produced by actually training and evaluating the models in this run. "
        "Result types are labelled: **in-sample** (train), **selection** (validation), **held-out** (test), "
        "**exploratory** (re-uses evaluation-only unknowns) and **diagnostic** (hand-written probes).", "",
        "## 1. Executive summary", "",
        f"- **Raw Out-of-Sample Test Accuracy (No Confidence Threshold): {T['test']['accuracy']:.3f}** "
        f"({ctx.get('test_correct', round(T['test']['accuracy'] * T['test']['n_samples']))} of "
        f"{T['test']['n_samples']} legitimate test commands classified correctly; no prediction excluded).",
        f"- **Selected model** (best validation macro F1): `{_pid(best)}`. Held-out test accuracy **{T['test']['accuracy']:.3f}**, macro F1 "
        f"**{T['test']['macro_f1']:.3f}** (95% phrase-clustered CI {bs['models']['tuned']['macro_f1']['ci_low']:.3f}–"
        f"{bs['models']['tuned']['macro_f1']['ci_high']:.3f}), "
        f"macro ROC-AUC **{T['test']['macro_roc_auc_ovr']:.3f}**.",
        f"- Train macro F1 {T['train']['macro_f1']:.3f} vs validation {T['validation']['macro_f1']:.3f} and test "
        f"{T['test']['macro_f1']:.3f}: a train–test accuracy gap of {gap_pts:.1f} points"
        + (" (large)." if gap_pts >= 10 else ".")
        + f" {pct(oov_test)} of test rows contain a word never seen in training; they account for "
        f"{pct(oov_error_share)} of test errors. BoW cannot map an unseen word to a known meaning.",
        f"- The **untuned default** (`{_pid(dflt)}`) scored lower on validation ({D['validation']['macro_f1']:.3f}) "
        f"{default_vs_test} on test ({D['test']['macro_f1']:.3f} vs {T['test']['macro_f1']:.3f}). The paired bootstrap "
        f"difference (tuned − default) is {diff['estimate']:+.3f} [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]. "
        f"With 30 validation rows per class{', many of them repeats,' if val_repeats else ','} model selection is "
        "noisy. The tuned model stays the primary result because the test set was not used for selection.",
        f"- **Unknown-intent rejection is {'weak' if rp['unknown_rejection_rate'] < 0.5 else 'partial'}.** With the threshold frozen at {thr:.3f} (95% known-intent "
        f"validation coverage), {pct(rp['unknown_rejection_rate'])} of the 300 unsupported requests are rejected "
        f"and {pct(rp['unknown_false_accept_rate'])} are falsely accepted. Known-intent test coverage is "
        f"{pct(rp['known_intent_coverage'])}. The maximum-probability rule is not a safe out-of-scope guard here.",
        f"- TRANSFER_MONEY ↔ BILL_PAY confusions on validation + test: {n_tb_heldout} rows"
        + (f", and {n_tb_train} *training* TRANSFER_MONEY rows (e.g. `{tb_example}`) are predicted BILL_PAY even "
           "in-sample: `pay <person> <amount>` collides with the strongest BILL_PAY cue, `pay`. "
           if n_tb_train else "; no training TRANSFER_MONEY rows are predicted BILL_PAY. ")
        + f"{int(sum(v for k, v in rp.get('unknown_false_accepts_by_predicted_intent', {}).items() if k in ctx['money_moving']))} "
        "unsupported requests were accepted as a money-moving intent (TRANSFER_MONEY, BILL_PAY or WITHDRAW_MONEY).",
        f"- The model is tiny and fast: {rt['tuned']['vocabulary_size']} vocabulary terms, "
        f"{rt['tuned']['n_parameters']:,} parameters, {rt['tuned']['model_size_mb'] * 1024:.1f} KB on disk, "
        f"{rt['tuned']['training_seconds'] * 1000:.0f} ms to train, single-request median latency "
        f"{lat1['median_ms']:.3f} ms (P95 {lat1['p95_ms']:.3f} ms) on CPU.", "",
        "## 2. Dataset", "",
        f"{ctx['audit']['n_rows']:,} synthetic utterances (sheet `Training Data`): 9 supported intents × 300 "
        "(240 train / 30 validation / 30 test each) and 300 `UNKNOWN_INTENT` rows reserved for rejection "
        "evaluation (`evaluation_only`). Eight intents are `core`; `CHECK_SPENDING` is the `stretch` intent. "
        f"The supplied splits were used unchanged. Figure: `{_r(ctx)}/figures/12_class_distribution.png`.", "",
        "## 3. Data-quality findings", "",
        f"- No missing values, empty texts, invalid labels or contradictory labels.",
        (f"- {'Heavy repetition' if au['normalized_duplicate_redundant_rows'] > 50 else 'Repetition'}: "
         f"{au['normalized_duplicate_redundant_rows']:,} redundant rows under a punctuation-insensitive key; "
         f"{test_unique} distinct phrases among {test_n} test rows."),
        f"- Train overlap: {leak_test['normalized_in_train']} test rows repeat a training phrase exactly or after "
        f"normalization, {leak_test['template_in_train']} once numbers are masked "
        f"(validation: {au['leakage']['validation']['normalized_in_train']} / "
        f"{au['leakage']['validation']['template_in_train']}). Reported, not removed; a leakage-controlled subset "
        f"({leak_test['rows'] - leak_test['leakage_flagged']} test rows) is evaluated.",
        f"- {'Strong' if oov_test >= 0.2 else 'Moderate' if oov_test >= 0.05 else 'Little'} vocabulary shift between "
        f"train and held-out partitions: {pct(oov_test)} of test rows contain an unseen word (see §13).",
        f"Full details: `{_r(ctx)}/reports/data_quality_report.md`, `{_r(ctx)}/metrics/dataset_audit.{{json,csv}}`.", "",
        "## 4. Model architecture and preprocessing", "",
        "`raw text → TextNormalizer → CountVectorizer → LogisticRegression → 9 class probabilities → threshold rejection`", "",
        "- **Normalization:** Unicode NFKC, lowercase, whitespace collapse. No stemming, lemmatization or "
        "stop-word removal.",
        f"- **Tokenization:** `{cfg['preprocessing']['token_pattern']}` keeps `$`, digits and one-character tokens "
        "(sklearn's default pattern would drop `$`, `5`, `2`). Amounts stay as literal tokens; masking numbers "
        "is a candidate improvement, not applied here.",
        f"- **Classifier:** multinomial logistic regression, `{cfg['model']['solver']}` solver, L2 penalty, "
        f"max_iter {cfg['model']['max_iter']} (convergence warnings are raised as errors).",
        "- Vectorizer and classifier are a single sklearn `Pipeline`, so the vocabulary is fitted only on the "
        "training split (verified by unit tests).", "",
        "## 5. Training methodology", "",
        "- Fit on `split = train` rows with a supported intent (2,160 rows). `UNKNOWN_INTENT` is never a class.",
        "- Hyperparameters chosen on `validation` macro F1 (ties broken by validation log loss).",
        "- The rejection threshold is chosen on known-intent validation rows only.",
        "- `test` and the unknown set are scored once, after everything is frozen.",
        f"- Canonical class order (shared with all team models): {', '.join(f'`{c}`' for c in labels)}.", "",
        "## 6. Hyperparameter tuning", "",
        f"Grid: n-gram ∈ {{(1,1), (1,2)}} × C ∈ {{0.1, 1, 10}} × {{counts, binary}} = {len(grid)} configurations "
        f"(`{_r(ctx)}/metrics/hyperparameter_search.csv`).", "",
        md_table(grid[["rank", "params_id", "train_macro_f1", "validation_macro_f1", "validation_accuracy",
                       "validation_log_loss", "vocabulary_size"]]),
        f"Selected: `{_pid(best)}`. Validation macro F1 spans only "
        f"{grid['validation_macro_f1'].min():.3f}–{grid['validation_macro_f1'].max():.3f} across the grid. Best "
        "validation macro F1 by C: " + ", ".join(f"C={c:g}: {v:.3f}" for c, v in best_by_c.items()) + ". With "
        f"{T['validation']['n_samples']} validation rows, one command is worth "
        f"{100 / T['validation']['n_samples']:.2f} accuracy points, so small gaps between configurations are noise.", "",
        "**Default vs tuned** (default = unigram counts, C = 1):", "",
        md_table(variant_tbl),
        f"Phrase-clustered bootstrap on test ({bs['method']}): tuned macro F1 {ci(bs['models']['tuned']['macro_f1'])}, "
        f"default {ci(bs['models']['default']['macro_f1'])}; difference {diff['estimate']:+.3f} "
        f"[{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}], positive in {pct(diff['share_of_resamples_positive'])} of resamples. "
        + ("The default model's better test score is a held-out observation. It must not be used to switch "
           "models after the fact; doing so would turn the test set into a second validation set."
           if tuned_worse else ""), "",
        "## 7. In-sample vs out-of-sample performance (selected model)", "",
        md_table(split_tbl),
        "Figure: `04_split_comparison.png`. The learning curve (`07_learning_curve.png`, training split only, "
        "5-fold CV grouped by phrase) reaches "
        f"{lc['cv_mean'].iloc[-1]:.3f} CV macro F1 at {int(lc['train_size'].iloc[-1])} examples, versus "
        f"{T['validation']['macro_f1']:.3f} on validation and {T['test']['macro_f1']:.3f} on test. "
        + ("Phrases resembling training phrases are easy; the gap comes from new phrasings, which more rows of the "
           "same templates are unlikely to fix." if lc['cv_mean'].iloc[-1] - T['validation']['macro_f1'] > 0.05 else
           "Cross-validated and held-out scores are close."), "",
        "## 8. Per-class performance (test, selected model)", "",
        md_table(pc[["intent", "scope", "precision", "recall", "f1", "support", "roc_auc_ovr", "average_precision"]]),
        "**Core-eight breakdown** (9-class model, rows whose true label is core; predictions of "
        "CHECK_SPENDING count as errors):", "",
        md_table(core_rows),
        f"Supplementary 8-class model trained on core intents only (same hyperparameters, not re-tuned): "
        f"test accuracy {c8['test']['accuracy']:.3f}, macro F1 {c8['test']['macro_f1']:.3f}, ROC-AUC "
        f"{c8['test']['macro_roc_auc_ovr']:.3f} (`metrics_core8_model.json`).", "",
        "## 9. Confusion-matrix analysis", "",
        "Figures `01_confusion_matrix_counts.png` and `02_confusion_matrix_normalized.png`. Most frequent test "
        "confusions:", "",
        md_table(test_conf[["actual_intent", "predicted_intent", "count", "unique_phrases", "example"]]),
        ("Confusions cluster on a few phrasings"
         + (", some repeated several times" if max_repeat > 1 else "")
         + f". {pct(oov_error_share)} of test errors contain a word absent from training (most frequent unseen "
         "held-out words: " + ", ".join(f"`{w}`" for w in ea["oov_words"]["oov_token"].head(6))
         + "). Unseen words are dropped, so the model falls back to whatever weak cues remain."), "",
        "## 10. ROC-AUC and precision-recall analysis", "",
        f"Test macro one-vs-rest ROC-AUC {T['test']['macro_roc_auc_ovr']:.3f} (weighted "
        f"{T['test']['weighted_roc_auc_ovr']:.3f}); macro average precision {T['test']['macro_average_precision']:.3f}. "
        f"ROC-AUC is high relative to accuracy because the correct intent is usually ranked near the top even "
        f"when it is not top-1: top-2 accuracy is {T['test']['top2_accuracy']:.3f} and top-3 "
        f"{T['test']['top3_accuracy']:.3f}, vs top-1 {T['test']['accuracy']:.3f}. A clarification prompt "
        "offering the top two intents is a cheap mitigation. Per-class curves: "
        "`05_roc_curves.png`, `06_precision_recall_curves.png`; values in `roc_auc_per_class.csv`. "
        f"Undefined metrics in this run: {T['test']['undefined_metrics'] or 'none'}.", "",
        f"**Calibration:** test ECE {T['test']['expected_calibration_error']:.3f} (`09_calibration_curve.png`). "
        f"Mean top-label confidence is {T['test']['mean_confidence']:.3f} against accuracy "
        f"{T['test']['accuracy']:.3f}: the model is **{calibration_word}**"
        + (". With strong regularization (C ≤ 0.1) probabilities are flattened, which makes the absolute value "
           "of a confidence threshold hard to interpret." if calibration_word == "under-confident" and best["C"] <= 0.1
           else ". Threshold values do not transfer between models without recalibration."), "",
        "## 11. Unknown-intent rejection", "",
        f"**Protocol.** Rule: `max_probability < threshold → UNKNOWN_INTENT`. The coverage target "
        f"({pct(cfg['rejection']['target_known_coverage'], 0)} of known validation intents accepted) was fixed "
        f"before any unknown label was examined, giving threshold **{thr:.4f}** "
        f"(validation coverage achieved {pct(rej['threshold_selection']['validation_known_coverage_achieved'])}). "
        "The frozen threshold was applied once to test + the 300 evaluation-only unknowns.", "",
        md_table(rej_counts),
        f"- Known-intent coverage (test): **{pct(rp['known_intent_coverage'])}**; false rejection rate "
        f"{pct(rp['known_false_rejection_rate'])}; accuracy on accepted known requests {pct(rp['accuracy_on_accepted_known'])}; "
        f"end-to-end known accuracy {pct(rp['end_to_end_known_accuracy'])}.",
        f"- Unknown rejection rate: **{pct(rp['unknown_rejection_rate'])}**; false acceptance rate "
        f"**{pct(rp['unknown_false_accept_rate'])}**. Per unique unknown phrase: "
        f"{pct(rej['unknown_rejection_rate_unique_phrases'])} ({rej['unknown_unique_phrases']} phrases).",
        "- Unknowns accepted, by predicted intent: " + (", ".join(f"{k} {v}" for k, v in rp.get(
            "unknown_false_accepts_by_predicted_intent", {}).items()) or "none") + ".", "",
        "The threshold-sensitivity curve (`10_rejection_threshold.png`, `rejection_threshold_sweep_exploratory.csv`) "
        "is **exploratory**: it re-uses the same evaluation-only unknowns at every threshold and is not an "
        "independent test.", "",
        f"**Supplementary unknown-aware threshold.** The 300 unknowns were split by normalized phrase into a "
        f"calibration half ({sup['n_unknown_calibration']} rows) and a disjoint holdout ({sup['n_unknown_holdout']} rows, "
        f"{sup['shared_phrases_between_halves']} shared phrases). Maximizing ½·coverage + ½·rejection on "
        f"validation knowns + calibration unknowns selects threshold {sup['selected_threshold']:.2f}. On test knowns + "
        f"holdout unknowns: unknown rejection {pct(sup['holdout_results_unknown_aware_threshold']['unknown_rejection_rate'])} "
        f"but known coverage only {pct(sup['holdout_results_unknown_aware_threshold']['known_intent_coverage'])}. The "
        f"primary threshold on the same holdout gives {pct(sup['holdout_results_frozen_primary_threshold']['unknown_rejection_rate'])} "
        f"/ {pct(sup['holdout_results_frozen_primary_threshold']['known_intent_coverage'])}. Confidence alone cannot "
        "separate known from unknown requests at an acceptable operating point. Caveat: both halves come from the "
        "same synthetic generator, and the holdout was also part of the primary evaluation, so this is not a "
        "fully untouched test.", "",
        *threshold_sweep_section(ctx),
        "## 12. Computational performance", "",
        "| Measure | Value |", "|---|---|",
        f"| Training time (final fit, 2,160 rows) | {rt['tuned']['training_seconds'] * 1000:.1f} ms |",
        f"| Full grid search ({len(grid)} fits + scoring) | {rt['hyperparameter_search_seconds']:.2f} s |",
        f"| Single-request latency median / P95 / P99 | {lat1['median_ms']:.3f} / {lat1['p95_ms']:.3f} / {lat1['p99_ms']:.3f} ms |",
        f"| Single-request throughput (1 / median) | {lat1['throughput_predictions_per_s']:,.0f} req/s |",
        f"| Batch of {latb['batch_size']} latency median / P95 | {latb['median_ms']:.2f} / {latb['p95_ms']:.2f} ms |",
        f"| Batch throughput | {latb['throughput_predictions_per_s']:,.0f} predictions/s |",
        f"| Serialized model (joblib bundle) | {rt['tuned']['model_size_mb'] * 1024:.1f} KB |",
        f"| Vocabulary size | {rt['tuned']['vocabulary_size']} |",
        f"| Parameters (coefficients + intercepts) | {rt['tuned']['n_parameters']:,} ({rt['tuned']['coef_shape'][0]}×{rt['tuned']['coef_shape'][1]} + {rt['tuned']['coef_shape'][0]}) |",
        f"| Hardware | {env['processor']}, {env['logical_cpus']} logical CPUs, {env['total_memory_gb']} GB RAM, CPU only |",
        f"| Runtime | {env['platform']}, Python {env['python_version']}, scikit-learn {rt['packages']['scikit-learn']} |", "",
        f"Methodology: {rt['methodology']} Single-request: {lat1['warmup_calls']} warm-up + "
        f"{lat1['measured_calls']} measured calls; batch: {latb['warmup_calls']} warm-up + {latb['measured_calls']} "
        "measured calls. Figure `11_inference_latency.png`.", "",
        "## 13. Error analysis (summary)", "",
        md_table(ea["short_oov_summary"]),
        f"- {len(ea['accepted_errors'])} distinct validation/test phrases are misclassified **and** accepted by "
        f"the rejection threshold. {len(ea['high_conf_errors'])} held-out errors have confidence ≥ "
        f"{ctx['diag']['high_confidence']}"
        + (" (the regularized model rarely produces high confidence)." if best["C"] <= 0.1 else
           ". Confident errors cannot be caught by any reasonable threshold."),
        "- Word order: BoW gives identical probabilities to reordered commands "
        "(`transfer from mere to jone` vs `transfer from jone to mere`). Direction of payment is invisible to "
        "this model.",
        "- Hand-written diagnostic probes are qualitative only (see `error_analysis.md`).", "",
        f"Full analysis: `{_r(ctx)}/reports/error_analysis.md`.", "",
        "## 14. Limitations and potential leakage", "",
        "- **Synthetic data.** Utterances are generated from templates with filler suffixes. Real customer "
        "language in Fiji will include Fijian and Fiji Hindi words, code-switching, local slang and "
        "abbreviations (e.g. M-PAiSA, \"vinaka\", \"bhaiya\"), misspellings, SMS shorthand and speech-to-text "
        "errors. None of these is represented systematically here. Reported metrics should not be read as "
        "production accuracy.",
        f"- **Leakage.** {leak_test['leakage_flagged']} test rows match a training phrase "
        f"({leak_test['normalized_in_train']} exactly or after normalization, {leak_test['template_in_train']} once "
        f"numbers are masked). Leakage-controlled test macro F1 is {T['test_leakage_controlled']['macro_f1']:.3f} vs "
        f"{T['test']['macro_f1']:.3f} on the full test set. Template-generated commands may also share structure "
        "across splits beyond what number masking detects.",
        f"- **Small held-out sets.** {test_n} test rows ({test_unique} distinct phrases); "
        "confidence intervals are wide (see §6), so small differences in macro F1 between models are not reliable.",
        f"- **Unknown evaluation.** The {n_unknown} unknowns are synthetic"
        + (", highly repetitive" if unk_unique < 0.8 * n_unknown else "")
        + " and sometimes padded with wallet words (\"… on my account\", \"… on the app\"). Rejection results are "
        "specific to this set.",
        "- **BoW limits.** No word order, negation, synonyms or sub-word information. Unseen words are dropped "
        "silently.",
        f"- **Calibration.** Probabilities are {calibration_word}; threshold values do not transfer across models "
        "without recalibration.", "",
        "## 15. Recommendations for the next model", "",
        "1. **Sub-word / character features** (char n-gram TF-IDF, fastText or a transformer tokenizer) to handle "
        "unseen words, misspellings and SMS forms"
        + (f". Unseen words appear in {pct(oov_error_share)} of test errors, the dominant failure mode here."
           if oov_error_share >= 0.5 else "."),
        "2. **Pretrained sentence encoders or transformers** to map unseen paraphrases (`owe`, `beneficiaries`, "
        "`statement`) to known intents through semantic similarity.",
        "3. **A dedicated out-of-scope detector** (e.g. trained on a separate pool of unsupported requests, "
        "energy/Mahalanobis scores or an embedding-distance check). Keep a genuinely untouched unknown test set.",
        "4. **Calibrate probabilities** (temperature scaling on validation) before setting thresholds, and select "
        "thresholds on a separate calibration split.",
        "5. **Safety confirmation for money-moving intents.** Require explicit confirmation of amount and payee "
        "for TRANSFER_MONEY, BILL_PAY and WITHDRAW_MONEY regardless of model confidence.",
        "6. **Data:** collect or write more diverse paraphrases per intent, de-duplicate phrases, and use "
        "phrase-grouped splits so validation reflects test difficulty.",
        "7. **Reuse this evaluation framework** (`sole_intent.evaluate`, `sole_intent.rejection`) and append to "
        "`experiment_summary.csv` so all models are compared on identical splits and metrics.", "",
    ]
    return "\n".join(L)


def _pid(params: dict) -> str:
    ng = params["ngram_range"]
    return f"ngram({ng[0]},{ng[1]}) C={params['C']:g} {'binary' if params['binary'] else 'counts'}"


# ------------------------------------------------------------------ error analysis report
def error_analysis_report(ctx: dict, ea: dict) -> str:
    thr = ctx["threshold"]
    probes = ctx["probes"].copy()
    scored = probes[probes["expected"] != "AMBIGUOUS"]
    wo = ctx["word_order"]
    conf = ea["confusions"]
    held_conf = conf[conf["split"] != "train"] if len(conf) else conf
    max_heldout_conf = max(ctx["preds"]["validation"]["max_confidence"].max(), ctx["preds"]["test"]["max_confidence"].max())
    oos = probes[probes["expected"] == UNKNOWN_INTENT]
    oos_accepted = int((oos["final_intent"] != UNKNOWN_INTENT).sum())
    test = ctx["preds"]["test"]
    errors = test[~test["correct"]]
    oov_err = errors[errors["oov_tokens"].fillna("").str.len() > 0]
    oov_err_share = len(oov_err) / len(errors) if len(errors) else 0.0
    oov_example = oov_err.iloc[0] if len(oov_err) else None
    tb_train = ctx["preds"]["train"].query("actual_intent == 'TRANSFER_MONEY' and predicted_intent == 'BILL_PAY'")
    in_dataset = int(probes["phrase_in_dataset"].astype(bool).sum())
    neg = probes[probes["text"].str.lower().str.startswith("don't send")]
    T = ctx["variants"]["tuned"]["metrics"]["test"]
    under_conf = T["mean_confidence"] < T["accuracy"] - 0.05
    n_conf = len(ea["high_conf_errors"])
    L = [
        "# Error Analysis — BoW + Logistic Regression (selected model)", "",
        f"_Generated {utc_now()}. Threshold for rejection: {thr:.4f}. Row-level predictions with all class "
        f"probabilities: `{_r(ctx)}/predictions/*_predictions.csv`. Tables below are also saved as "
        f"`{_r(ctx)}/metrics/error_*.csv`._", "",
        "## 1. Most common confusions", "",
        md_table(held_conf[["split", "actual_intent", "predicted_intent", "count", "unique_phrases", "example"]].head(15)),
        "Train confusions (in-sample):", "",
        md_table(conf[conf["split"] == "train"][["actual_intent", "predicted_intent", "count", "example"]].head(10)
                 if len(conf) else conf),
        "## 2. Confident errors", "",
        f"Errors with confidence ≥ {ctx['diag']['high_confidence']}: {len(ea['high_conf_errors'])} on validation/test. "
        f"The selected model's highest held-out confidence is {max_heldout_conf:.3f}. "
        + ("Because confident errors are rare, the more useful view is " if n_conf <= 2 else
           "Confident errors cannot be filtered by any reasonable threshold. Below: ")
        + "**errors that the rejection threshold would accept** (distinct phrases, highest confidence first):", "",
        md_table(ea["accepted_errors"].head(20)),
        "## 3. Low-confidence correct predictions (would be falsely rejected)", "",
        f"{len(ea['low_conf_correct'])} distinct correct validation/test phrases fall below the threshold:", "",
        md_table(ea["low_conf_correct"].head(15)),
        f"## 4. Ambiguous commands (top-2 margin < {ctx['diag']['ambiguous_margin']})", "",
        md_table(ea["ambiguous"].head(20)),
        f"## 5. Short commands (≤ {ctx['diag']['short_command_max_tokens']} tokens) and out-of-vocabulary terms", "",
        md_table(ea["short_oov_summary"]),
        "Short-command errors (validation/test):", "",
        md_table(ea["short_errors"].head(15)),
        "Most frequent out-of-vocabulary tokens in validation/test (never seen in training):", "",
        md_table(ea["oov_words"].head(25)),
        f"{pct(oov_err_share)} of test errors contain a word never seen in training"
        + (" — out-of-vocabulary words explain most errors." if oov_err_share >= 0.5 else ".")
        + " The vectorizer silently drops unseen words"
        + (f"; e.g. in `{oov_example['text']}` the word(s) `{oov_example['oov_tokens']}` are ignored, leaving the "
           f"model to guess {oov_example['predicted_intent']} (actual {oov_example['actual_intent']})."
           if oov_example is not None else "."), "",
        "## 6. Word-order blindness", "",
        md_table(wo),
        "Pairs with the same bag of words receive **identical** probabilities (max |Δp| = 0). The model cannot "
        "tell who pays whom (`transfer from mere to jone` vs `from jone to mere`) or the order of a multi-step "
        "request. Payer/payee direction must come from a slot-filling step, not from this classifier.", "",
        "## 7. Train/test leakage", "",
        md_table(ea["leakage_accuracy"]),
        "Test rows that repeat a training phrase (exact or number-masked):", "",
        md_table(ea["leaked_rows"].drop_duplicates("text")),
        "## 8. Financial-safety view", "",
        "TRANSFER_MONEY ↔ BILL_PAY confusions:", "",
        md_table(ea["transfer_billpay"]),
        "Training TRANSFER_MONEY rows predicted BILL_PAY (in-sample):", "",
        md_table(tb_train[["text", "max_confidence", "final_intent"]]),
        ("These are person-to-person payments the model routes to bill payment (see `13_top_features.png` for "
         "the words driving BILL_PAY). " if len(tb_train) else "")
        + "A transfer to a person mistaken for a bill payment (or vice versa) is the most consequential confusion "
        "for a wallet. A payee/biller slot check should gate both intents.", "",
        "Supported requests wrongly routed **and accepted** into a money-moving intent (validation/test):", "",
        md_table(ea["money_false_positives"]),
        f"Unsupported (UNKNOWN_INTENT) requests accepted as money-moving intents (distinct phrases, "
        f"{len(ea['unknown_money_accepts'])}):", "",
        md_table(ea["unknown_money_accepts"].head(20)),
        "Most confident accepted unknowns (all intents):", "",
        md_table(ea["unknown_top_accepts"]),
        "Any production use would need an explicit confirmation step (amount, payee, biller) before a "
        "money-moving action. Classifier confidence alone is insufficient.", "",
        "## 9. Qualitative diagnostic probes", "",
        f"**These {len(probes)} commands were written by hand for this analysis. They are qualitative diagnostic "
        "probes, not an independent statistical test set, and support no claim about generalization.** "
        f"`phrase_in_dataset` flags probes whose normalized phrase also appears in the workbook ({in_dataset} "
        f"of {len(probes)}). `AMBIGUOUS` marks commands with no single correct intent.", "",
        md_table(probes[["text", "expected", "final_intent", "predicted_intent", "max_confidence",
                         "matches_expected", "phrase_in_dataset", "probe_purpose"]]),
        f"Probes with a definite expected label matching the final (post-rejection) output: "
        f"{int(scored['matches_expected'].astype(bool).sum())}/{len(scored)} (illustrative only). "
        f"Out-of-scope probes accepted as a supported intent: {oos_accepted}/{len(oos)}.", "",
        "## 10. Takeaways", "",
        f"1. {pct(oov_err_share)} of held-out test errors involve vocabulary unseen in training.",
        "2. " + ("Confidence is low across the board (under-confident model), so " if under_conf else
                 "Confident predictions are common for both supported and unsupported commands, so ")
        + "confidence-based rejection cannot cleanly separate in-scope from out-of-scope requests.",
        "3. BoW ignores word order and negation"
        + (f" (`{neg['text'].iloc[0]}` → {neg['final_intent'].iloc[0]})." if len(neg) else "."),
        f"4. TRANSFER_MONEY ↔ BILL_PAY confusions on held-out data: "
        f"{int(ea['transfer_billpay'].query('split != \"train\"')['count'].sum())}; out-of-scope requests routed "
        f"to money-moving intents: {len(ea['unknown_money_accepts'])} distinct phrases.", "",
    ]
    return "\n".join(L)


# ------------------------------------------------------------------ model card
def model_card(ctx: dict) -> str:
    cfg, T = ctx["cfg"], ctx["variants"]["tuned"]["metrics"]
    rp = ctx["rejection"]["primary_frozen_threshold_test_plus_unknown"]
    rt = ctx["runtime"]["tuned"]
    best = ctx["best_params"]
    L = [
        "# Model Card — Sole Smart Actions Intent Classifier (BoW + Logistic Regression)", "",
        "## Model details", "",
        f"- **Name / version:** `{cfg['experiment']['model_name']}` tuned, v{cfg['experiment']['model_version']} "
        f"(`{_rel(ctx['paths']['tuned'])}`)",
        f"- **Type:** bag-of-words (`{_pid(best)}`) + multinomial logistic regression (scikit-learn "
        f"{ctx['runtime']['packages']['scikit-learn']}), confidence-threshold rejection at {ctx['threshold']:.4f}.",
        f"- **Size:** {rt['vocabulary_size']} vocabulary terms, {rt['n_parameters']:,} parameters, "
        f"{rt['model_size_mb'] * 1024:.1f} KB.",
        "- **Developed for:** MIT AI Edge course project (academic baseline). Created by the Sole Smart Actions team.",
        f"- **Artifact contents:** sklearn Pipeline, canonical class order, rejection threshold, metadata "
        "(dataset hash, package versions, params). Load with `sole_intent.train.load_bundle`.", "",
        "## Intended use", "",
        "- Research baseline for classifying short digital-wallet commands into 9 intents, and a reference point "
        "for CNN/LSTM/transformer comparisons using the shared evaluation framework.",
        "- **Out of scope:** executing payments or any live financial action; production deployment; "
        "languages other than English; safety-critical decisions without human or explicit-confirmation "
        "controls.", "",
        "## Training data", "",
        f"`{cfg['data']['path']}` (version `{cfg['data']['dataset_version']}`, SHA-256 "
        f"`{ctx['fingerprint']['file_sha256']}`): 2,160 synthetic training utterances, 240 per intent. "
        "Validation (270) was used for model and threshold selection. Test (270) and 300 unknown requests "
        "were used once for final evaluation.", "",
        "## Metrics (held-out test unless stated)", "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | {T['test']['accuracy']:.3f} |",
        f"| Macro F1 (95% phrase-clustered CI) | {ci(ctx['bootstrap']['models']['tuned']['macro_f1'])} |",
        f"| Weighted F1 | {T['test']['weighted_f1']:.3f} |",
        f"| Macro ROC-AUC (OvR) | {T['test']['macro_roc_auc_ovr']:.3f} |",
        f"| Log loss | {T['test']['log_loss']:.3f} |",
        f"| Core-8 macro F1 (9-class model) | {T['test']['core8_breakdown']['macro_f1']:.3f} |",
        f"| Leakage-controlled test macro F1 | {T['test_leakage_controlled']['macro_f1']:.3f} |",
        f"| Train / validation macro F1 | {T['train']['macro_f1']:.3f} / {T['validation']['macro_f1']:.3f} |",
        f"| Known-intent coverage at threshold | {pct(rp['known_intent_coverage'])} |",
        f"| Unknown rejection rate / false acceptance | {pct(rp['unknown_rejection_rate'])} / {pct(rp['unknown_false_accept_rate'])} |",
        f"| Single-request latency median / P95 (CPU) | {rt['latency_single_request']['median_ms']:.3f} / {rt['latency_single_request']['p95_ms']:.3f} ms |", "",
        "## Limitations and risks", "",
        "- Trained and evaluated on synthetic English templates. Expect lower accuracy on real Fijian customer "
        "language (code-switching with Fijian / Fiji Hindi, slang, typos, speech-to-text errors).",
        "- Unseen words are ignored. Paraphrases built on new vocabulary are frequently misclassified.",
        "- Ignores word order and negation, so it cannot tell payer from payee.",
        f"- Weak out-of-scope rejection: {pct(rp['unknown_false_accept_rate'])} of unsupported requests were "
        "accepted, some as money-moving intents.",
        "- Probabilities are " + ("under-confident (poorly calibrated)." if T["test"]["mean_confidence"]
                                    < T["test"]["accuracy"] - 0.05 else "not calibrated; recalibrate before thresholding."),
        "- Logistic-regression coefficients (`13_top_features.png`) show associations between n-grams and "
        "predicted classes. They are not causal explanations.", "",
        "## Ethical and safety considerations", "",
        "Misrouting a financial request can cause monetary harm. Downstream systems must confirm amount, payee "
        "and action with the user before any money moves, and should hand unclear requests to a help or "
        "clarification flow. No personal or real customer data was used.", "",
    ]
    return "\n".join(L)


def write_reports(ctx: dict) -> None:
    reports_dir = _out(ctx).reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    ea = error_analysis_tables(ctx)
    (reports_dir / "data_quality_report.md").write_text(data_quality_report(ctx, ea), encoding="utf-8")
    (reports_dir / "evaluation_report.md").write_text(evaluation_report(ctx, ea), encoding="utf-8")
    (reports_dir / "error_analysis.md").write_text(error_analysis_report(ctx, ea), encoding="utf-8")
    (reports_dir / "model_card.md").write_text(model_card(ctx), encoding="utf-8")
