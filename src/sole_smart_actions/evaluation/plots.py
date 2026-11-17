"""Static report figures (PNG). One fixed colour per model, recessive grid, single y-axis, direct labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Validated categorical slots 1-4 (blue, orange, aqua, yellow); colour follows the model, never its rank.
SHORT_NAMES = {"bow": "Bag of Words", "tfidf": "TF-IDF", "transformer": "Custom Transformer", "distilbert": "DistilBERT"}
MODEL_COLORS = {"bow": "#2a78d6", "tfidf": "#eb6834", "transformer": "#1baf7a", "distilbert": "#eda100"}
INTENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DPI = 160


def _style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(SURFACE)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def confusion_matrix_png(cm: np.ndarray, labels: list[str], title: str, path: Path) -> Path:
    cm = np.asarray(cm)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 6.8))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(labels)), labels, fontsize=8, color=MUTED)
    ax.set_xlabel("Predicted intent", color=MUTED)
    ax.set_ylabel("True intent", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    return _save(fig, path)


def roc_curves_png(y_true, proba: np.ndarray, labels: list[str], title: str, path: Path) -> Path:
    y_true = np.asarray(y_true, dtype=object)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    for i, c in enumerate(labels):
        pos = y_true == c
        if pos.any() and (~pos).any():
            fpr, tpr, _ = roc_curve(pos, proba[:, i])
            ax.plot(fpr, tpr, lw=1.6, color=INTENT_COLORS[i], label=c)
    ax.set_xlabel("False positive rate", color=MUTED)
    ax.set_ylabel("True positive rate", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, None)
    return _save(fig, path)


def threshold_sweep_png(dense: pd.DataFrame, default: float, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    series = [("known_coverage", "Legitimate commands accepted", "#2a78d6"),
              ("overall_correct_routing_rate", "Legitimate commands routed correctly", "#1baf7a"),
              ("unknown_safe_fallback_rate", "Unknown commands safely handled", "#eb6834"),
              ("unknown_to_financial_rate", "Unknown commands sent to a money flow", "#e34948")]
    for col, label, color in series:
        ax.plot(dense["threshold"], dense[col], lw=2, color=color, label=label)
    ax.axvline(default, color=MUTED, ls="--", lw=1)
    ax.text(default + 0.01, 0.45, f"validation default\n{default:.3f}", color=MUTED, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Confidence threshold (max probability)", color=MUTED)
    ax.set_ylabel("Share of commands", color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    _style(ax)
    return _save(fig, path)


def model_bars_png(df: pd.DataFrame, value_col: str, title: str, ylabel: str, path: Path, *, fmt="{:.1%}",
                   ylim=None, log=False) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(df))
    colors = [MODEL_COLORS.get(m, MUTED) for m in df["model"]]
    bars = ax.bar(x, df[value_col], width=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, df[value_col]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(x, [SHORT_NAMES.get(m, m) for m in df["model"]], fontsize=9, color=MUTED)
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _style(ax)
    return _save(fig, path)


def grouped_bars_png(df: pd.DataFrame, cols: list[tuple[str, str]], title: str, ylabel: str, path: Path,
                     fmt="{:.0%}") -> Path:
    """Metrics on x, one bar per model (model colours), legend for models."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    n = len(df)
    w = 0.8 / n
    x = np.arange(len(cols))
    for k, (_, r) in enumerate(df.iterrows()):
        vals = [r[c] for c, _ in cols]
        bars = ax.bar(x + (k - (n - 1) / 2) * w, vals, w, color=MODEL_COLORS.get(r["model"], MUTED),
                      edgecolor=SURFACE, linewidth=1.5, label=r["display_name"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(x, [lab for _, lab in cols], fontsize=9, color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style(ax)
    return _save(fig, path)
