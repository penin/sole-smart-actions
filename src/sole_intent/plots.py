"""Figure generation. Each function saves a PNG and returns the data behind it as a DataFrame
(so it can also be written to CSV). Model-agnostic: inputs are labels, probabilities and
class order, not a specific estimator.

Styling: one recessive grid, thin marks, fixed categorical hue order (never cycled). Per-intent
curves use 3x3 small multiples so nine intents never need nine hues.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_CMAP = sns.blend_palette(["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"], as_cmap=True)
DPI = 200


def apply_style(dpi: int = 200) -> None:
    global DPI
    DPI = dpi
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "lines.linewidth": 2, "text.color": INK,
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _short(label: str) -> str:
    return label.replace("_", " ").title()


def confusion_matrix_plot(cm: pd.DataFrame, path: Path, title: str, normalized: bool) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalized else "d", cmap=SEQ_CMAP, cbar=True,
                linewidths=1, linecolor=SURFACE, vmin=0, vmax=1 if normalized else None,
                annot_kws={"size": 8}, ax=ax, square=True)
    ax.set_xticklabels([_short(c) for c in cm.columns], rotation=40, ha="right")
    ax.set_yticklabels([_short(c) for c in cm.index], rotation=0)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("Actual intent")
    ax.set_title(title, loc="left")
    ax.grid(False)
    _save(fig, path)
    return cm


def per_class_metrics_plot(per_class: pd.DataFrame, path: Path, title: str) -> pd.DataFrame:
    data = per_class[["intent", "precision", "recall", "f1"]].copy()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(data))
    width = 0.26
    for i, metric in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (i - 1) * width, data[metric], width * 0.92, color=SERIES[i], label=metric.title())
    ax.set_xticks(x, [_short(c) for c in data["intent"]], rotation=30, ha="right")
    lo = max(0.0, np.nanmin(data[["precision", "recall", "f1"]].to_numpy(dtype=float)) - 0.1)
    ax.set_ylim(lo, 1.01)
    ax.set_ylabel("Score")
    ax.set_title(title, loc="left")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.grid(axis="x", visible=False)
    _save(fig, path)
    return data


def split_comparison_plot(table: pd.DataFrame, path: Path, title: str) -> pd.DataFrame:
    """table: index = metric names, columns = splits."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    metrics, splits = list(table.index), list(table.columns)
    x = np.arange(len(metrics))
    width = 0.8 / len(splits)
    for i, split in enumerate(splits):
        vals = table[split].to_numpy(dtype=float)
        bars = ax.bar(x + (i - (len(splits) - 1) / 2) * width, vals, width * 0.92, color=SERIES[i], label=split)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK_2, rotation=90)
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics], rotation=20, ha="right")
    lo = max(0.0, np.nanmin(table.to_numpy(dtype=float)) - 0.08)
    ax.set_ylim(lo, 1.06)
    ax.set_ylabel("Score")
    ax.set_title(title, loc="left")
    ax.legend(ncol=len(splits), loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.grid(axis="x", visible=False)
    _save(fig, path)
    return table


def _small_multiples():
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
    return fig, axes.ravel()


def roc_curves_plot(y_true, proba, class_order: list[str], path: Path, title: str) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=object)
    rows = []
    fig, axes = _small_multiples()
    for i, (label, ax) in enumerate(zip(class_order, axes)):
        pos = y_true == label
        if pos.all() or not pos.any():
            ax.text(0.5, 0.5, "undefined\n(single class)", ha="center", color=MUTED)
            continue
        fpr, tpr, thr = roc_curve(pos, proba[:, i])
        area = auc(fpr, tpr)
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
        ax.plot(fpr, tpr, color=SERIES[0])
        ax.set_title(f"{_short(label)}  AUC {area:.3f}", fontsize=9, loc="left")
        rows.append(pd.DataFrame({"intent": label, "fpr": fpr, "tpr": tpr, "threshold": thr}))
    for ax in axes[6:]:
        ax.set_xlabel("False positive rate")
    for ax in axes[::3]:
        ax.set_ylabel("True positive rate")
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pr_curves_plot(y_true, proba, class_order: list[str], path: Path, title: str) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=object)
    rows = []
    fig, axes = _small_multiples()
    for i, (label, ax) in enumerate(zip(class_order, axes)):
        pos = y_true == label
        if not pos.any():
            ax.text(0.5, 0.5, "undefined\n(no positives)", ha="center", color=MUTED)
            continue
        prec, rec, thr = precision_recall_curve(pos, proba[:, i])
        ap = average_precision_score(pos, proba[:, i])
        ax.axhline(pos.mean(), color=AXIS, lw=1, ls="--")
        ax.plot(rec, prec, color=SERIES[0], drawstyle="steps-post")
        ax.set_title(f"{_short(label)}  AP {ap:.3f}", fontsize=9, loc="left")
        rows.append(pd.DataFrame({"intent": label, "recall": rec, "precision": prec,
                                  "threshold": np.append(thr, np.nan)}))
    for ax in axes[6:]:
        ax.set_xlabel("Recall")
    for ax in axes[::3]:
        ax.set_ylabel("Precision")
    fig.suptitle(title + "  (dashed = class prevalence)", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def learning_curve_plot(lc: pd.DataFrame, path: Path, title: str) -> pd.DataFrame:
    """lc columns: train_size, train_mean, train_std, cv_mean, cv_std."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col, color, name in (("train", SERIES[0], "Training folds"), ("cv", SERIES[1], "CV held-out folds")):
        ax.fill_between(lc["train_size"], lc[f"{col}_mean"] - lc[f"{col}_std"],
                        lc[f"{col}_mean"] + lc[f"{col}_std"], color=color, alpha=0.15, lw=0)
        ax.plot(lc["train_size"], lc[f"{col}_mean"], color=color, marker="o", ms=5, label=name)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Macro F1")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    _save(fig, path)
    return lc


def confidence_distribution_plot(conf: np.ndarray, correct: np.ndarray, threshold: float,
                                 path: Path, title: str) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(0, 1, 26)
    for mask, color, name in ((correct, SERIES[0], "Correct"), (~correct, SERIES[1], "Incorrect")):
        ax.hist(conf[mask], bins=bins, color=color, alpha=0.75, label=f"{name} (n={int(mask.sum())})",
                edgecolor=SURFACE, linewidth=1)
    ax.axvline(threshold, color=INK_2, ls="--", lw=1.2)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f" threshold {threshold:.3f}", color=INK_2, fontsize=8, va="top")
    ax.set_xlabel("Maximum predicted probability")
    ax.set_ylabel("Examples")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    _save(fig, path)
    return pd.DataFrame({"max_confidence": conf, "correct": correct})


def calibration_plot(bins: pd.DataFrame, ece: float, path: Path, title: str) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1, label="Perfect calibration")
    valid = bins[bins["count"] > 0]
    ax.plot(valid["mean_confidence"], valid["accuracy"], color=SERIES[0], marker="o", ms=6,
            label=f"Model (ECE {ece:.3f})")
    for _, r in valid.iterrows():
        ax.annotate(f"n={int(r['count'])}", (r["mean_confidence"], r["accuracy"]), textcoords="offset points",
                    xytext=(4, -10), fontsize=7, color=MUTED)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean predicted confidence (top label)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    _save(fig, path)
    return bins


def rejection_threshold_plot(sweep: pd.DataFrame, threshold: float, path: Path, title: str) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(sweep["threshold"], sweep["known_intent_coverage"], color=SERIES[0], label="Known-intent coverage (test)")
    ax.plot(sweep["threshold"], sweep["accuracy_on_accepted_known"], color=SERIES[2],
            label="Accuracy on accepted known (test)")
    ax.plot(sweep["threshold"], sweep["unknown_rejection_rate"], color=SERIES[1],
            label="Unknown rejection rate (evaluation-only, exploratory)")
    ax.axvline(threshold, color=INK_2, ls="--", lw=1.2)
    ax.text(threshold, 0.03, f" frozen threshold {threshold:.3f}\n (95% validation coverage)", color=INK_2, fontsize=8)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.03)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.12))
    _save(fig, path)
    return sweep


def latency_plot(samples: dict[str, np.ndarray], path: Path, title: str) -> pd.DataFrame:
    fig, axes = plt.subplots(1, len(samples), figsize=(5.5 * len(samples), 4.2))
    axes = np.atleast_1d(axes)
    frames = []
    for i, ((name, ms), ax) in enumerate(zip(samples.items(), axes)):
        upper = np.percentile(ms, 99.5)
        ax.hist(ms[ms <= upper], bins=50, color=SERIES[i], edgecolor=SURFACE, linewidth=0.5)
        for q, ls in ((50, "-"), (95, "--"), (99, ":")):
            v = np.percentile(ms, q)
            ax.axvline(v, color=INK_2, ls=ls, lw=1)
            ax.text(v, ax.get_ylim()[1] * (0.95 - 0.1 * [50, 95, 99].index(q)), f" P{q} {v:.2f}",
                    fontsize=7.5, color=INK_2, va="top")
        ax.set_xlabel("Latency per call (ms)")
        ax.set_ylabel("Calls")
        ax.set_title(name, loc="left", fontsize=10)
        frames.append(pd.DataFrame({"measurement": name, "latency_ms": ms}))
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return pd.concat(frames, ignore_index=True)


def class_distribution_plot(counts: pd.DataFrame, path: Path, title: str) -> pd.DataFrame:
    """counts: index = intent, columns = splits."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(counts))
    width = 0.8 / len(counts.columns)
    for i, split in enumerate(counts.columns):
        ax.bar(x + (i - (len(counts.columns) - 1) / 2) * width, counts[split], width * 0.92,
               color=SERIES[i], label=split)
    ax.set_xticks(x, [_short(c) for c in counts.index], rotation=30, ha="right")
    ax.set_ylabel("Examples")
    ax.set_title(title, loc="left")
    ax.legend(ncol=len(counts.columns))
    ax.grid(axis="x", visible=False)
    _save(fig, path)
    return counts


def top_features_plot(features: pd.DataFrame, class_order: list[str], path: Path, title: str) -> pd.DataFrame:
    """features columns: intent, rank, feature, coefficient."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for label, ax in zip(class_order, axes.ravel()):
        sub = features[features["intent"] == label].sort_values("coefficient")
        ax.barh(sub["feature"], sub["coefficient"], color=SERIES[0], height=0.75)
        ax.set_title(_short(label), loc="left", fontsize=10)
        ax.tick_params(axis="y", labelsize=7.5, colors=INK_2)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("LR coefficient", fontsize=8)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return features


# ------------------------------------------------------------------ threshold analysis (exploratory)
def _mark_thresholds(ax, baseline: float, candidate: float | None, label: bool = True) -> None:
    marks = [(baseline, "V1 0.196", "--")]
    if candidate is not None:
        marks.append((candidate, f"candidate {candidate:.2f}", ":"))
    for x, name, ls in marks:
        ax.axvline(x, color=INK_2, ls=ls, lw=1)
        if label:
            ax.text(x, 1.0, f" {name}", transform=ax.get_xaxis_transform(), fontsize=7, color=INK_2,
                    va="top", rotation=90, ha="right")


def threshold_panels(table: pd.DataFrame, baseline: float, candidate: float | None, path: Path,
                     ref_name: str = "V1 frozen threshold 0.196",
                     title: str = "Threshold comparison: Baseline V1 model (exploratory; test + evaluation-only unknowns)",
                     ) -> pd.DataFrame:
    """Small multiples: one metric per panel, each vs threshold (single y-scale per panel)."""
    panels = [("unknown_rejection_rate", "Unknown rejection rate", True),
              ("known_coverage", "Known-intent coverage", True),
              ("accepted_accuracy", "Accuracy on accepted known commands", True),
              ("safe_fallback_rate", "Safe fallback rate (unknowns)", True),
              ("unknown_to_financial", "Unknowns routed to financial actions", False),
              ("end_to_end_known_accuracy", "End-to-end known accuracy\n(rejections count as wrong)", True)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    t = table["threshold"].to_numpy()
    for (col, panel_title, is_rate), ax in zip(panels, axes.ravel()):
        y = table[col].to_numpy(dtype=float)
        ax.plot(t, y, color=SERIES[0], marker="o", ms=5)
        for xi, yi in zip(t, y):
            ax.annotate(f"{yi:.0%}" if is_rate else f"{yi:.0f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=INK_2)
        _mark_thresholds(ax, baseline, candidate, label=False)
        ax.set_title(panel_title, loc="left", fontsize=10)
        if is_rate:
            ax.set_ylim(0, 1.08)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.set_ylim(0, max(1.0, np.nanmax(y)) * 1.18)
    for ax in axes[1]:
        ax.set_xlabel("Confidence threshold")
    key = f"dashed line = {ref_name}" + (f" · dotted line = candidate {candidate:.2f}" if candidate is not None else "")
    fig.suptitle(f"{title}\n{key}",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return table


def coverage_vs_rejection(table: pd.DataFrame, dense: pd.DataFrame, baseline: float, candidate: float | None,
                          path: Path, ref_short: str = "0.196 (V1)") -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(dense["unknown_rejection_rate"], dense["known_intent_coverage"], color=AXIS, lw=1.5,
            label="All thresholds 0.00–1.00 (step 0.01)")
    ax.plot(table["unknown_rejection_rate"], table["known_coverage"], ls="none", marker="o", ms=7,
            color=SERIES[0], label="Evaluated thresholds", markeredgecolor=SURFACE, markeredgewidth=1.5)
    for _, r in table.iterrows():
        name = ref_short if r["threshold"] == baseline else ("none" if r["threshold"] == 0 else f"{r['threshold']:.2f}")
        bold = r["threshold"] in (baseline, candidate)
        ax.annotate(name, (r["unknown_rejection_rate"], r["known_coverage"]), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color=INK if bold else INK_2, fontweight="bold" if bold else "normal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Unknown commands rejected")
    ax.set_ylabel("Legitimate commands accepted (coverage)")
    ax.set_title("Coverage vs unknown rejection trade-off (exploratory)", loc="left")
    ax.legend(loc="lower left")
    _save(fig, path)
    return table


def safe_fallback_comparison(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    """100% stacked bars: what happens to unsupported commands at each threshold."""
    parts = [("unknown_rejected", "Rejected → fallback (help/clarify)", SERIES[0]),
             ("unknown_to_help_support", "Classified HELP_SUPPORT → routed to help", SERIES[2]),
             ("unknown_to_other_intents", "Routed to other non-financial intent", SERIES[3]),
             ("unknown_to_financial", "Routed to financial action", SERIES[1])]
    n = table["unknown_n"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for col, name, color in parts:
        share = table[col].to_numpy(dtype=float) / n
        ax.bar(x, share, 0.7, bottom=bottom, color=color, label=name, edgecolor=SURFACE, linewidth=1.5)
        bottom += share
    for xi, v in zip(x, table["safe_fallback_rate"]):
        ax.text(xi, 1.02, f"safe {v:.0%}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, table["label"].str.replace(" [candidate]", "\n[candidate]", regex=False), fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Share of 300 unsupported commands")
    ax.set_xlabel("Confidence threshold")
    ax.set_title("Unsupported commands: application outcome by threshold (exploratory)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)
    _save(fig, path)
    return table


def financial_routing_risk(table: pd.DataFrame, financial: list[str], path: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for i, intent in enumerate(financial):
        vals = table[f"unknown_to_{intent}"].to_numpy(dtype=float)
        ax.bar(x, vals, 0.7, bottom=bottom, color=SERIES[i], label=_short(intent), edgecolor=SURFACE, linewidth=1.5)
        bottom += vals
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + 0.4, f"{tot:.0f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, table["label"].str.replace(" [candidate]", "\n[candidate]", regex=False), fontsize=8)
    ax.set_ylim(0, max(bottom.max() * 1.2, 1))
    ax.set_ylabel("Unsupported commands (of 300)")
    ax.set_xlabel("Confidence threshold")
    ax.set_title("Unsupported commands accepted as money-moving intents (exploratory)", loc="left")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    _save(fig, path)
    return table


def legitimate_rejection_heatmap(by_intent: pd.DataFrame, labels: list[str], path: Path) -> pd.DataFrame:
    rate = by_intent.pivot(index="intent", columns="threshold", values="false_rejection_rate").reindex(labels)
    count = by_intent.pivot(index="intent", columns="threshold", values="false_rejected").reindex(labels)
    annot = rate.map(lambda v: f"{v:.0%}") + count.map(lambda v: f"\n({v:.0f})")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(rate, annot=annot, fmt="", cmap=SEQ_CMAP, vmin=0, vmax=1, linewidths=1, linecolor=SURFACE,
                annot_kws={"size": 7.5}, cbar_kws={"format": matplotlib.ticker.PercentFormatter(1.0)}, ax=ax)
    ax.set_xticklabels(["none" if t == 0 else f"{t:.3f}".rstrip("0") if t < 0.2 else f"{t:.2f}"
                        for t in rate.columns])
    ax.set_yticklabels([_short(c) for c in rate.index], rotation=0)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Actual intent (30 test commands each)")
    ax.set_title("Legitimate commands rejected, by intent: rate (count)", loc="left")
    ax.grid(False)
    _save(fig, path)
    return rate
