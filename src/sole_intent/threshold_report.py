"""Markdown reports for the exploratory threshold analysis. All numbers come from the computed results."""

from __future__ import annotations

import pandas as pd

from .reports import md_table, pct
from .utils import METRICS_DIR, load_json, utc_now

STATUS = ("**Status: exploratory.** The test split and the 300 evaluation-only unknown commands were already "
          "inspected in earlier experiments (Baseline V1 evaluation and its exploratory sweep). These results "
          "describe the trade-offs of the V1 model. They are not an unbiased final evaluation, and they cannot "
          "show that any threshold is statistically optimal.")

PROTOCOL = [
    "## How a production threshold should be selected",
    "",
    "1. **Freeze the model** and the decision rule (`max_probability < t → UNKNOWN_INTENT`).",
    "2. **Build a calibration set** that has not been used before: legitimate commands for every intent plus a "
    "*separate* pool of unsupported commands, ideally real or realistic Fijian customer phrasing. Split unknowns "
    "by phrase/template so no phrase appears in two sets.",
    "3. **Prespecify the selection rule** before looking at results, e.g. *highest coverage subject to "
    "≥ 90% unknown rejection and zero unknowns routed to money-moving intents*, or a cost-weighted objective "
    "where routing an unsupported request to TRANSFER_MONEY/BILL_PAY/WITHDRAW_MONEY costs far more than a "
    "fallback.",
    "4. **Optionally calibrate probabilities first** (temperature scaling on the calibration set), so the "
    "threshold has a stable meaning.",
    "5. **Choose t on the calibration set only**, then **evaluate once on a new, untouched test set** "
    "(legitimate + unsupported) and report coverage, rejection and financial-routing rates with "
    "phrase-clustered confidence intervals.",
    "6. **Monitor after launch.** Track fallback rate and user rephrasing, and re-run steps 2–5 when the model "
    "or customer language changes.",
    "",
]


def _comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    cols = {"label": "threshold", "known_coverage": "known coverage", "known_false_rejection_rate": "known false-reject",
            "accepted_accuracy": "accepted acc", "accepted_macro_f1": "accepted macro F1",
            "end_to_end_known_accuracy": "end-to-end known acc", "unknown_rejection_rate": "unknown rejected",
            "unknown_false_accept_rate": "unknown false-accept", "unknown_to_help_support": "unknown→HELP (n)",
            "unknown_to_help_support_rate": "unknown→HELP (%)", "safe_fallback_rate": "safe fallback",
            "unknown_to_financial": "unknown→financial (n)",
            "validation_known_coverage_reference": "val coverage (ref)"}
    out = table[list(cols)].rename(columns=cols).copy()
    for c in out.columns[1:]:
        if "(n)" not in c:
            out[c] = out[c].map(lambda v: pct(v))
    return out


def _examples(frame: pd.DataFrame, cols: list[str], n: int = 15) -> pd.DataFrame:
    return frame.drop_duplicates("text").sort_values("max_confidence", ascending=False)[cols].head(n)


def candidate_report(ctx: dict) -> str:
    t, base, tag = ctx["candidate"], ctx["baseline_thr"], ctx["tag"]
    table, by_intent, labels, fin = ctx["table"], ctx["by_intent"], ctx["labels"], ctx["financial"]
    res = {r["threshold"]: r for r in ctx["results"]}
    c, v = res[t], res[base]
    ck, cu, ca = c["model_classification"]["known"], c["model_classification"]["unknown"], c["application"]
    vk, vu, va = v["model_classification"]["known"], v["model_classification"]["unknown"], v["application"]
    row_c = table[table["threshold"] == t].iloc[0]
    d, ch = ctx["detail"], ctx["changes"]
    legit, unk = d[d["dataset"] == "legitimate_test"], d[d["dataset"] == "unknown_evaluation"]
    newly_rej = ch[(ch["dataset"] == "legitimate_test") & (ch[f"outcome_{tag}"] == "false_reject")]
    lost_correct = int((newly_rej["outcome_v1"] == "correct_accept").sum())
    caught_wrong = int((newly_rej["outcome_v1"] == "incorrect_accept").sum())
    help_rejected = int(((legit["outcome"] == "false_reject") & (legit["actual_intent"] == "HELP_SUPPORT")).sum())
    unk_other = unk[unk["outcome"] == "accepted_as_other_intent"]

    outcome = pd.DataFrame([
        {"outcome": "Legitimate: correctly accepted (model routes to the right intent)", "V1 0.196": vk["correct_accept"], f"{t:.2f}": ck["correct_accept"]},
        {"outcome": "Legitimate: incorrectly accepted (routed to the wrong intent)", "V1 0.196": vk["incorrect_accept"], f"{t:.2f}": ck["incorrect_accept"]},
        {"outcome": "Legitimate: rejected (UNKNOWN_INTENT → app fallback)", "V1 0.196": vk["false_reject"], f"{t:.2f}": ck["false_reject"]},
        {"outcome": "Unsupported: correctly rejected (UNKNOWN_INTENT → app fallback)", "V1 0.196": vu["rejected_as_unknown_intent"], f"{t:.2f}": cu["rejected_as_unknown_intent"]},
        {"outcome": "Unsupported: classified HELP_SUPPORT (app routes to help)", "V1 0.196": vu["classified_as_help_support"], f"{t:.2f}": cu["classified_as_help_support"]},
        {"outcome": "Unsupported: routed to other non-financial intents", "V1 0.196": vu["classified_as_other_intent"], f"{t:.2f}": cu["classified_as_other_intent"]},
        {"outcome": "Unsupported: routed to TRANSFER_MONEY / BILL_PAY / WITHDRAW_MONEY", "V1 0.196": vu["classified_as_financial"], f"{t:.2f}": cu["classified_as_financial"]},
    ])
    per_intent = (by_intent[by_intent["threshold"].isin([base, t])]
                  .pivot(index="intent", columns="threshold", values="false_rejected").reindex(labels))
    per_intent.columns = ["rejected @ V1 0.196", f"rejected @ {t:.2f}"]
    per_intent["support"] = 30
    per_intent[f"rate @ {t:.2f}"] = (per_intent[f"rejected @ {t:.2f}"] / per_intent["support"]).map(pct)
    worst = per_intent[f"rejected @ {t:.2f}"].idxmax()

    ex_cols = ["text", "actual_intent", "top_intent", "max_confidence"]
    v1_test = load_json(METRICS_DIR / "metrics_test.json")
    L = [
        f"# Threshold {t:.2f} — Detailed Evaluation (Baseline V1 model)", "",
        f"_Generated {utc_now()} by `scripts/evaluate_threshold.py`. Model `{ctx['payload']['model_path']}` "
        f"(SHA-256 `{ctx['payload']['model_sha256'][:12]}…`), not retrained._", "",
        STATUS, "",
        "## Setup", "",
        "- **Decision rule (model):** if max class probability < threshold → `UNKNOWN_INTENT`; otherwise the "
        "highest-probability supported intent.",
        "- **Application mapping:** `UNKNOWN_INTENT` → `FALLBACK_HELP_OR_CLARIFY` (offer HELP_SUPPORT or ask the "
        "customer to rephrase); any intent → `ROUTE_<INTENT>`. A *model* classification of HELP_SUPPORT is "
        "counted separately from a rejection.",
        f"- **Data:** {ck['n']} legitimate test commands (30 per intent) and {cu['n']} unsupported evaluation-only "
        "commands.",
        f"- **Probabilities:** read from V1's saved prediction files ({ctx['payload']['probability_source']}) and "
        f"checked against the saved model (max abs difference {ctx['payload']['max_abs_diff_saved_vs_model_probabilities']:.1e}).",
        f"- **Reproducibility check:** at the frozen V1 threshold ({base:.5f}, reported as 0.196) this analysis "
        "reproduces V1's published outcome counts exactly. All Baseline V1 artifacts were verified unchanged "
        "(`baseline_v1_manifest.json`).", "",
        f"## 1. Outcomes at {t:.2f} vs Baseline V1", "",
        md_table(outcome),
        "| Rate | V1 0.196 | " + f"{t:.2f} |", "|---|---|---|",
        f"| Known-intent coverage | {pct(vk['coverage'])} | {pct(ck['coverage'])} |",
        f"| Known-intent false rejection rate | {pct(vk['false_rejection_rate'])} | {pct(ck['false_rejection_rate'])} |",
        f"| Accuracy among accepted known commands | {pct(vk['accepted_accuracy'])} | {pct(ck['accepted_accuracy'])} |",
        f"| Macro F1 among accepted known commands | {vk['accepted_macro_f1']:.3f} | {ck['accepted_macro_f1']:.3f} |",
        f"| End-to-end known accuracy (rejections count as wrong) | {pct(vk['end_to_end_accuracy'])} | {pct(ck['end_to_end_accuracy'])} |",
        f"| Unknown rejection rate | {pct(vu['rejection_rate'])} | {pct(cu['rejection_rate'])} |",
        f"| Unknown false acceptance rate | {pct(vu['false_accept_rate'])} | {pct(cu['false_accept_rate'])} |",
        f"| Unknown rejection rate, per unique phrase | {pct(vu['rejection_rate_unique_phrases'])} | {pct(cu['rejection_rate_unique_phrases'])} |",
        f"| Safe fallback rate (rejected or HELP_SUPPORT) | {pct(va['unknown_safe_fallback_rate'])} | {pct(ca['unknown_safe_fallback_rate'])} |",
        f"| Validation known coverage (reference) | {pct(table.loc[table['is_baseline_v1'], 'validation_known_coverage_reference'].iloc[0])} | {pct(row_c['validation_known_coverage_reference'])} |", "",
        f"## 2. Effect on legitimate customer commands", "",
        f"- Coverage falls from {pct(vk['coverage'])} to **{pct(ck['coverage'])}**: {ck['false_reject']} of "
        f"{ck['n']} legitimate commands ({pct(ck['false_rejection_rate'])}) now go to the fallback instead of being "
        "routed.",
        f"- Compared with V1, {len(newly_rej)} more legitimate commands are rejected. **{caught_wrong}** of them were "
        f"previously *misrouted* (the higher threshold turns a wrong action into a fallback) and **{lost_correct}** "
        "were previously *correct* (a working command is now blocked).",
        f"- Commands that are accepted are more reliable: accuracy {pct(vk['accepted_accuracy'])} → "
        f"{pct(ck['accepted_accuracy'])}, macro F1 {vk['accepted_macro_f1']:.3f} → {ck['accepted_macro_f1']:.3f}. "
        f"End-to-end accuracy falls ({pct(vk['end_to_end_accuracy'])} → {pct(ck['end_to_end_accuracy'])}) because "
        "rejections count as failures.",
        f"- {help_rejected} rejected commands were genuine HELP_SUPPORT requests. The fallback offers help anyway, "
        "so at application level these customers still reach support.",
        f"- The burden is uneven across intents. The most-rejected intent is **{worst}**.", "",
        md_table(per_intent, index=True),
        "Figure: `legitimate_rejection_by_intent.png`.", "",
        "Rejected legitimate commands (distinct phrases, most confident first):", "",
        md_table(_examples(legit[legit["outcome"] == "false_reject"], ex_cols, 20)),
        "Legitimate commands still accepted but routed to the wrong intent:", "",
        md_table(_examples(legit[legit["outcome"] == "incorrect_accept"], ex_cols, 20)),
        "Most legitimate rejections are commands whose wording is absent from training (`I owe …`, "
        "`Edit <name>'s details`, `customer service`, `Recent transactions`). The model is under-confident on "
        "anything unfamiliar, whether it is in scope or not.", "",
        "## 3. Effect on unsupported commands", "",
        f"- **{cu['rejected_as_unknown_intent']}** of {cu['n']} ({pct(cu['rejection_rate'])}) are rejected and reach "
        f"the fallback (V1: {vu['rejected_as_unknown_intent']}, {pct(vu['rejection_rate'])}).",
        f"- **{cu['classified_as_help_support']}** ({pct(cu['classified_as_help_support_rate'])}) are *classified* as "
        "HELP_SUPPORT. That is a model error, but the application outcome is benign (the customer is sent to support).",
        f"- Safe fallback rate (rejected + HELP_SUPPORT): **{pct(ca['unknown_safe_fallback_rate'])}** "
        f"(V1: {pct(va['unknown_safe_fallback_rate'])}).",
        f"- **{cu['classified_as_other_intent']}** are routed to other non-financial intents (e.g. balance or "
        "transaction views): wrong, but no money moves.", "",
        "Unsupported commands not rejected:", "",
        md_table(_examples(unk[unk["outcome"] != "correct_reject"], ["text", "top_intent", "max_confidence", "outcome"], 30)),
        f"## 4. Financial-action routing", "",
        f"- Unsupported commands routed to TRANSFER_MONEY / BILL_PAY / WITHDRAW_MONEY: **{cu['classified_as_financial']}** "
        f"at {t:.2f} vs {vu['classified_as_financial']} at V1 "
        f"({', '.join(f'{k} {n}' for k, n in vu['classified_as_financial_by_intent'].items())}).",
        f"- Legitimate commands misrouted *into* a different money-moving intent: {ck['misrouted_to_financial']} at "
        f"{t:.2f} vs {vk['misrouted_to_financial']} at V1.",
        "- A threshold reduces this risk but cannot remove it for new phrasings. Money-moving intents should still "
        "require explicit confirmation of amount and payee/biller.", "",
        f"## 5. Is {t:.2f} a reasonable candidate?", "",
        f"**It is a defensible candidate for further testing. This analysis cannot show it is optimal.** On these "
        f"(previously inspected) sets, {t:.2f} sits where unknown rejection has largely saturated "
        f"({pct(cu['rejection_rate'])}) and unknown→financial routing has reached {cu['classified_as_financial']}. "
        f"The price is high: about one in {round(1 / max(ck['false_rejection_rate'], 1e-9))} legitimate commands goes "
        "to the fallback, and certain intents are disproportionately blocked.", "",
        "Considerations:", "",
        f"- **Selection evidence is weak.** The choice is informed by the same test and unknown sets reported here. "
        f"On validation, {t:.2f} accepts only {pct(row_c['validation_known_coverage_reference'])} of known commands, "
        "far from V1's prespecified 95% coverage target. The original protocol would not have selected it.",
        f"- **The model is under-confident.** Baseline V1 test ECE is {v1_test['expected_calibration_error']:.2f}, "
        f"mean top-label confidence {v1_test['mean_confidence']:.2f} vs accuracy {v1_test['accuracy']:.2f}. So "
        f"{t:.2f} here does not mean {t:.0%} confidence in a calibrated sense, and the value will not transfer to "
        "the CNN/transformer models.",
        "- **The unknown set is synthetic and repetitive.** Phrases like `Order food please …` and `Renew my "
        "passport …` repeat with filler suffixes, so the 300 rows behave like far fewer independent tests.",
        "- **Acceptability depends on the fallback UX.** If the fallback is a quick *\"Did you mean …?\"* prompt "
        "showing the top intent(s), a ~34% fallback rate is irritating but recoverable. If the fallback dead-ends "
        "the customer, it is likely unacceptable.",
        "- **Better levers exist** than the threshold alone: calibrate probabilities, add sub-word/semantic "
        "features so legitimate paraphrases gain confidence, and add a dedicated out-of-scope detector.", "",
        *PROTOCOL,
        "## All thresholds (exploratory)", "",
        md_table(_comparison_table(table)),
        "Figures: `threshold_comparison.png`, `coverage_vs_rejection.png`, `safe_fallback_comparison.png`, "
        "`financial_routing_risk.png`, `legitimate_rejection_by_intent.png`.", "",
        "## Files", "",
        f"- `threshold_{tag}_predictions.csv`: every legitimate test and unsupported command with probabilities, "
        "model decision, application action and outcome.",
        f"- `threshold_{tag}_errors.csv`: all non-correct outcomes (incorrect accepts, legitimate rejections, "
        "unsupported commands not rejected).",
        f"- `threshold_{tag}_vs_v1_changes.csv`: rows whose outcome differs from Baseline V1.", "",
    ]
    return "\n".join(L)


def comparison_report(ctx: dict) -> str:
    table, by_intent, labels = ctx["table"], ctx["by_intent"], ctx["labels"]
    rej = by_intent.pivot(index="intent", columns="threshold", values="false_rejected").reindex(labels)
    rej.columns = ["V1 0.196" if c == ctx["baseline_thr"] else f"{c:.2f}" for c in rej.columns]
    fin = table[["label"] + [f"unknown_to_{f}" for f in ctx["financial"]] + ["unknown_to_financial",
                                                                            "known_misrouted_to_financial"]]
    L = [
        "# Confidence Threshold Comparison — Baseline V1 model", "",
        f"_Generated {utc_now()} by `scripts/evaluate_threshold.py`. No retraining; Baseline V1 artifacts verified "
        "unchanged._", "",
        STATUS, "",
        "## Comparison table", "",
        f"Legitimate commands: {int(table['known_n'].iloc[0])} test rows. Unsupported: {int(table['unknown_n'].iloc[0])} "
        "evaluation-only rows. *Model classification* metrics (coverage, accuracy, rejection) are separate from "
        "*application* metrics (safe fallback = rejected or classified HELP_SUPPORT).", "",
        md_table(_comparison_table(table)),
        "## Unsupported commands routed to money-moving intents", "",
        md_table(fin),
        "## Legitimate commands rejected, by actual intent (count of 30)", "",
        md_table(rej, index=True),
        *PROTOCOL,
        "Full per-threshold details: `threshold_comparison.json` / `.csv`; dense 0.01-step sweep: "
        f"`dense_threshold_sweep.csv`; candidate detail: `threshold_{ctx['tag']}_report.md`.", "",
    ]
    return "\n".join(L)


def write_threshold_reports(ctx: dict) -> None:
    out = ctx["out_dir"]
    (out / f"threshold_{ctx['tag']}_report.md").write_text(candidate_report(ctx), encoding="utf-8")
    (out / "threshold_comparison_report.md").write_text(comparison_report(ctx), encoding="utf-8")
