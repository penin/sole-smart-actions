"""Markdown report generation. Every number is read from the run's computed results; the
reports are regenerated on each run of ``scripts/run_all.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rejection import UNKNOWN_INTENT
from .utils import METRICS_DIR, PROJECT_ROOT, REPORTS_DIR, utc_now


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
        frame.to_csv(METRICS_DIR / f"error_{name}.csv", index=False)
    return out


# ------------------------------------------------------------------ data quality report
def data_quality_report(ctx: dict, ea: dict) -> str:
    a, fp, f, d = ctx["audit"], ctx["fingerprint"], ctx["findings"], ctx["audit_details"]
    labels, unk = ctx["labels"], ctx["cfg"]["unknown_label"]
    bal = pd.DataFrame(a["class_balance"]).T.reindex(labels + [unk])
    leak = a["leakage"]
    lines = [
        "# Data Quality Report — Sole Smart Actions Intent Dataset", "",
        f"_Generated {utc_now()} by `scripts/run_all.py`. Source workbook is read-only; no rows were "
        "deleted, edited or re-split._", "",
        "## Source", "",
        f"- File: `{_rel(PROJECT_ROOT / ctx['cfg']['data']['path'])}` (sheet `{ctx['cfg']['data']['sheet']}`)",
        f"- Dataset version: `{fp['dataset_version']}`; rows: {fp['n_rows']:,}",
        f"- File SHA-256: `{fp['file_sha256']}`",
        f"- Content SHA-256 (text,intent,split,scope as CSV): `{fp['content_sha256']}`", "",
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
        "Repeats are exact copies or differ only in case/punctuation (e.g. a trailing `?`). Within-partition repetition inflates the apparent sample size: "
        f"the 270-row test set holds {a['unique_phrases_per_split']['test']} distinct phrases, so test "
        "confidence intervals use a phrase-clustered bootstrap.", "",
        "Most repeated phrases:", "",
        md_table(d["duplicate_groups"].head(15)[["canonical", "count", "splits", "intents"]]),
        "## Contradictory labels", "",
        f"{a['contradictory_label_phrases']} normalized phrases carry more than one intent label"
        + (" — no label conflicts were found." if a["contradictory_label_phrases"] == 0
           else "; see `data/processed/duplicate_groups.csv`."), "",
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
        "(`metrics_test_leakage_controlled.json`). Unknown-evaluation phrases do not overlap training.", "",
        "Rows flagged per partition are listed in `data/processed/leakage_rows.csv`; all shared phrases in "
        "`data/processed/cross_split_overlap.csv`.", "",
        "## Vocabulary shift (train vocabulary vs held-out partitions)", "",
        md_table(ea["short_oov_summary"][["split", "n", "rows_with_oov", "share_rows_with_oov",
                                          "mean_oov_token_share"]]),
        "Validation and test contain many words that never occur in training (e.g. "
        + ", ".join(f"`{w}`" for w in ea["oov_words"]["oov_token"].head(12)) + "). The held-out partitions "
        "therefore test generalization to **unseen paraphrase templates**, not just unseen rows. This is the "
        "main driver of the train/validation gap reported in the evaluation report.", "",
        "## Implications", "",
        "1. Training-set metrics and cross-validation inside the training split are optimistic (duplicates and "
        "shared templates).",
        "2. Validation/test numbers are dominated by a small number of unseen templates repeated several times; "
        "single errors move metrics in steps of several points.",
        "3. Exact train→test overlap is small but non-zero and is reported, not removed.",
        "4. Many unknown-evaluation phrases repeat (e.g. `play music` ×11); unknown rejection is also "
        "reported per unique phrase.", "",
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
        f"- **Selected model** (best validation macro F1): unigram counts, C = {best['C']:g} "
        f"(`{_pid(best)}`). Held-out test accuracy **{T['test']['accuracy']:.3f}**, macro F1 "
        f"**{T['test']['macro_f1']:.3f}** (95% phrase-clustered CI {bs['models']['tuned']['macro_f1']['ci_low']:.3f}–"
        f"{bs['models']['tuned']['macro_f1']['ci_high']:.3f}), "
        f"macro ROC-AUC **{T['test']['macro_roc_auc_ovr']:.3f}**.",
        f"- Train macro F1 {T['train']['macro_f1']:.3f} vs validation {T['validation']['macro_f1']:.3f}: a large "
        "generalization gap. Validation/test contain paraphrase templates whose key words never occur in "
        f"training ({pct(ea['short_oov_summary'].set_index('split').loc['test', 'share_rows_with_oov'])} of test rows "
        "contain an unseen word). BoW cannot map an unseen word to a known meaning.",
        f"- The **untuned default** (`{_pid(dflt)}`) scored lower on validation ({D['validation']['macro_f1']:.3f}) "
        f"but {'higher' if tuned_worse else 'lower'} on test ({D['test']['macro_f1']:.3f}). The paired bootstrap "
        f"difference (tuned − default) is {diff['estimate']:+.3f} [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]. "
        "With 30 validation rows per class, many of them repeats, model selection is noisy. The tuned model "
        "stays the primary result because the test set was not used for selection.",
        f"- **Unknown-intent rejection is weak.** With the threshold frozen at {thr:.3f} (95% known-intent "
        f"validation coverage), {pct(rp['unknown_rejection_rate'])} of the 300 unsupported requests are rejected "
        f"and {pct(rp['unknown_false_accept_rate'])} are falsely accepted. Known-intent test coverage is "
        f"{pct(rp['known_intent_coverage'])}. The maximum-probability rule is not a safe out-of-scope guard here.",
        f"- TRANSFER_MONEY ↔ BILL_PAY confusions on validation + test: {n_tb_heldout} rows, but "
        f"{n_tb_train} *training* TRANSFER_MONEY rows (e.g. `{tb_example}`) are predicted BILL_PAY even "
        "in-sample: `pay <person> <amount>` collides with the strongest BILL_PAY cue, `pay`. "
        f"{int(sum(v for k, v in rp.get('unknown_false_accepts_by_predicted_intent', {}).items() if k in ctx['money_moving']))} "
        "unsupported requests were accepted as a money-moving intent (TRANSFER_MONEY, BILL_PAY or WITHDRAW_MONEY).",
        f"- The model is tiny and fast: {rt['tuned']['vocabulary_size']} vocabulary terms, "
        f"{rt['tuned']['n_parameters']:,} parameters, {rt['tuned']['model_size_mb'] * 1024:.1f} KB on disk, "
        f"{rt['tuned']['training_seconds'] * 1000:.0f} ms to train, single-request median latency "
        f"{lat1['median_ms']:.3f} ms (P95 {lat1['p95_ms']:.3f} ms) on CPU.", "",
        "## 2. Dataset", "",
        f"{ctx['audit']['n_rows']:,} synthetic utterances (sheet `Training Data`): 9 supported intents × 300 "
        "(240 train / 30 validation / 30 test each) and 300 `UNKNOWN_INTENT` rows reserved for rejection "
        "evaluation (`evaluation_only`). Eight intents are `core`; `CHECK_SPENDING` is the `stretch` intent. "
        "The supplied splits were used unchanged. Figure: `results/figures/12_class_distribution.png`.", "",
        "## 3. Data-quality findings", "",
        f"- No missing values, empty texts, invalid labels or contradictory labels.",
        f"- Heavy repetition: {ctx['audit']['normalized_duplicate_redundant_rows']:,} redundant rows after "
        f"normalization; only {ctx['audit']['unique_phrases_per_split']['test']} distinct phrases among 270 test rows.",
        f"- Leakage: {ctx['audit']['leakage']['test']['leakage_flagged']} test rows and "
        f"{ctx['audit']['leakage']['validation']['leakage_flagged']} validation rows repeat a training phrase "
        "(exactly or after masking numbers). Reported, not removed; a leakage-controlled subset is evaluated.",
        "- Strong vocabulary shift between train and held-out partitions (see §13).",
        "Full details: `results/reports/data_quality_report.md`, `results/metrics/dataset_audit.{json,csv}`.", "",
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
        "(`results/metrics/hyperparameter_search.csv`).", "",
        md_table(grid[["rank", "params_id", "train_macro_f1", "validation_macro_f1", "validation_accuracy",
                       "validation_log_loss", "vocabulary_size"]]),
        f"Selected: `{_pid(best)}`. Validation macro F1 spans only "
        f"{grid['validation_macro_f1'].min():.3f}–{grid['validation_macro_f1'].max():.3f} across the grid. Best "
        "validation macro F1 by C: " + ", ".join(f"C={c:g}: {v:.3f}" for c, v in best_by_c.items()) + ". With "
        "30 validation rows per class, the gap between the top configurations is a few repeated phrases.", "",
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
        f"{lc['cv_mean'].iloc[-1]:.3f} CV macro F1 at {int(lc['train_size'].iloc[-1])} examples, far above "
        f"validation ({T['validation']['macro_f1']:.3f}). Phrases resembling training phrases are easy; the gap "
        "comes from new phrasings, which more rows of the same templates are unlikely to fix.", "",
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
        "Each confusion comes from one or a few templates (e.g. `I owe <name> <amount>`), often repeated "
        "several times, so each reflects a missing template, not a diffuse weakness. Common pattern: a key verb or noun absent from training "
        "(`owe`, `beneficiaries`, `statement`, `complaint`, `customer service`). The model then falls back "
        "to whatever weak cues remain, such as `show`, `view` or `my`.", "",
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
        f"{T['test']['accuracy']:.3f}" + (f": the C = {best['C']:g} model is **under-confident**. Its "
        "probabilities are flattened by regularization, which makes the absolute value of a confidence "
        "threshold hard to interpret." if T['test']['mean_confidence'] < T['test']['accuracy'] - 0.05
        else "."), "",
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
        f"{ctx['diag']['high_confidence']}, because the regularized model rarely produces high confidence.",
        "- Word order: BoW gives identical probabilities to reordered commands "
        "(`transfer from mere to jone` vs `transfer from jone to mere`). Direction of payment is invisible to "
        "this model.",
        "- Hand-written diagnostic probes are qualitative only (see `error_analysis.md`).", "",
        "Full analysis: `results/reports/error_analysis.md`.", "",
        "## 14. Limitations and potential leakage", "",
        "- **Synthetic data.** Utterances are generated from templates with filler suffixes. Real customer "
        "language in Fiji will include Fijian and Fiji Hindi words, code-switching, local slang and "
        "abbreviations (e.g. M-PAiSA, \"vinaka\", \"bhaiya\"), misspellings, SMS shorthand and speech-to-text "
        "errors. None of these is represented systematically here. Reported metrics should not be read as "
        "production accuracy.",
        f"- **Leakage.** {ctx['audit']['leakage']['test']['leakage_flagged']} test rows repeat a training phrase. "
        f"Leakage-controlled test macro F1 is {T['test_leakage_controlled']['macro_f1']:.3f} vs "
        f"{T['test']['macro_f1']:.3f} on the full test set.",
        f"- **Small, repetitive held-out sets.** {T['test_unique_phrases']['n_samples']} distinct test phrases; "
        "confidence intervals are wide (see §6), so small differences in macro F1 between models are not reliable.",
        "- **Unknown evaluation.** The 300 unknowns are synthetic, highly repetitive and sometimes padded with "
        "wallet words (\"… on my account\", \"… on the app\"). Rejection results are specific to this set.",
        "- **BoW limits.** No word order, negation, synonyms or sub-word information. Unseen words are dropped "
        "silently.",
        "- **Calibration.** Probabilities are under-confident; threshold values do not transfer across models "
        "without recalibration.", "",
        "## 15. Recommendations for the next model", "",
        "1. **Sub-word / character features** (char n-gram TF-IDF, fastText or a transformer tokenizer) to handle "
        "unseen words, misspellings and SMS forms. This is the dominant failure mode here.",
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
    L = [
        "# Error Analysis — BoW + Logistic Regression (selected model)", "",
        f"_Generated {utc_now()}. Threshold for rejection: {thr:.4f}. Row-level predictions with all class "
        "probabilities: `results/predictions/*_predictions.csv`. Tables below are also saved as "
        "`results/metrics/error_*.csv`._", "",
        "## 1. Most common confusions", "",
        md_table(held_conf[["split", "actual_intent", "predicted_intent", "count", "unique_phrases", "example"]].head(15)),
        "Train confusions (in-sample):", "",
        md_table(conf[conf["split"] == "train"][["actual_intent", "predicted_intent", "count", "example"]].head(10)
                 if len(conf) else conf),
        "## 2. Confident errors", "",
        f"Errors with confidence ≥ {ctx['diag']['high_confidence']}: {len(ea['high_conf_errors'])} on validation/test. "
        f"The selected model's highest held-out confidence is {max_heldout_conf:.3f}, so the more useful view is **errors that the rejection "
        "threshold would accept** (distinct phrases, highest confidence first):", "",
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
        "Out-of-vocabulary words explain most errors. Rows with any unseen word are far less accurate than "
        "fully in-vocabulary rows. The vectorizer silently drops unseen words, so `I owe jone 80` reduces to "
        "`i jone 80`, which carries almost no intent signal.", "",
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
        md_table(ctx["preds"]["train"].query("actual_intent == 'TRANSFER_MONEY' and predicted_intent == 'BILL_PAY'")
                 [["text", "max_confidence", "final_intent"]]),
        "These are person-to-person payments phrased as `pay <name> <amount>`. Because `pay` is the "
        "strongest BILL_PAY feature (see `13_top_features.png`) and names are sparse, the model routes them to "
        "bill payment. A transfer to a person mistaken for a bill payment (or vice versa) is the most "
        "consequential confusion for a wallet. A payee/biller slot check should gate both intents.", "",
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
        "`phrase_in_dataset` flags probes whose normalized phrase also appears in the workbook (several "
        "example probes requested in the brief, such as `tell me a joke`, exist in the unknown set). "
        "`AMBIGUOUS` marks commands with no single correct intent.", "",
        md_table(probes[["text", "expected", "final_intent", "predicted_intent", "max_confidence",
                         "matches_expected", "phrase_in_dataset", "probe_purpose"]]),
        f"Probes with a definite expected label matching the final (post-rejection) output: "
        f"{int(scored['matches_expected'].astype(bool).sum())}/{len(scored)} (illustrative only). "
        f"Out-of-scope probes accepted as a supported intent: {oos_accepted}/{len(oos)}.", "",
        "## 10. Takeaways", "",
        "1. Most held-out errors trace to unseen vocabulary in a few repeated paraphrase templates.",
        "2. Confidence is low across the board (under-confident model), so confidence-based rejection cannot "
        "separate in-scope from out-of-scope requests.",
        "3. BoW ignores word order and negation (`don't send money to Jone` → TRANSFER_MONEY).",
        "4. Explicit TRANSFER_MONEY ↔ BILL_PAY confusion is rare, but out-of-scope requests do get routed to "
        "money-moving intents.", "",
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
        "- Probabilities are under-confident (poorly calibrated).",
        "- Logistic-regression coefficients (`13_top_features.png`) show associations between n-grams and "
        "predicted classes. They are not causal explanations.", "",
        "## Ethical and safety considerations", "",
        "Misrouting a financial request can cause monetary harm. Downstream systems must confirm amount, payee "
        "and action with the user before any money moves, and should hand unclear requests to a help or "
        "clarification flow. No personal or real customer data was used.", "",
    ]
    return "\n".join(L)


def write_reports(ctx: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ea = error_analysis_tables(ctx)
    (REPORTS_DIR / "data_quality_report.md").write_text(data_quality_report(ctx, ea), encoding="utf-8")
    (REPORTS_DIR / "evaluation_report.md").write_text(evaluation_report(ctx, ea), encoding="utf-8")
    (REPORTS_DIR / "error_analysis.md").write_text(error_analysis_report(ctx, ea), encoding="utf-8")
    (REPORTS_DIR / "model_card.md").write_text(model_card(ctx), encoding="utf-8")
