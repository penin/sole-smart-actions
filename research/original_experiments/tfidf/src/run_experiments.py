"""
Sole Smart Actions - intent classification experiments (Project 1, 15.071 The AI Edge)

Question: does a better text representation (TF-IDF, with and without character
n-grams) classify wallet commands better than the Bag-of-Words benchmark?

Run from the repository root:
    python src/run_experiments.py

Everything is deterministic (fixed seeds) and uses the predefined V5 splits:
    train (2,160) -> fit models
    validation (270) -> choose hyperparameters and the confidence threshold
    test (270) + evaluation_only UNKNOWN_INTENT (300) -> reported once, at the end
Outputs are written to results/.
"""
from __future__ import annotations

import json
import pickle
import random
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx"
OUT = ROOT / "results"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

UNKNOWN = "UNKNOWN_INTENT"
MONEY_MOVING = {"TRANSFER_MONEY", "BILL_PAY", "WITHDRAW_MONEY"}
TARGET_COVERAGE = 0.95  # threshold keeps >=95% of valid in-scope commands (set on validation)
C_GRID = [0.5, 1, 2, 5, 10, 20, 50]


# --------------------------------------------------------------------------- data
def load_data():
    df = pd.read_excel(DATA, sheet_name="Training Data")
    df["text"] = df["text"].astype(str).str.strip()
    parts = {s: df[df.split == s].reset_index(drop=True)
             for s in ["train", "validation", "test", "evaluation_only"]}
    return df, parts


# --------------------------------------------------------------------------- models
def make_lr(C):
    return LogisticRegression(C=C, max_iter=5000, random_state=SEED)


def build(name, params):
    if name == "BoW + LR":
        vec = CountVectorizer(lowercase=True, ngram_range=params["ngram"])
    elif name == "TF-IDF (word) + LR":
        vec = TfidfVectorizer(lowercase=True, ngram_range=params["ngram"],
                              sublinear_tf=True)
    elif name == "TF-IDF (word + char) + LR":
        vec = FeatureUnion([
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb",
                                     ngram_range=params["char"], sublinear_tf=True)),
        ])
    else:
        raise ValueError(name)
    return Pipeline([("vec", vec), ("clf", make_lr(params["C"]))])


SEARCH_SPACE = {
    "BoW + LR": [{"ngram": (1, 1), "C": c} for c in C_GRID],  # team benchmark: unigram counts
    "TF-IDF (word) + LR": [{"ngram": g, "C": c} for g in [(1, 1), (1, 2)] for c in C_GRID],
    "TF-IDF (word + char) + LR": [{"char": g, "C": c}
                                  for g in [(2, 4), (2, 5), (3, 5)] for c in C_GRID],
}


# --------------------------------------------------------------- business-rule benchmark
RULES = [  # ordered keyword rules: a stand-in for today's "search the menu" behaviour
    ("HELP_SUPPORT", r"\b(help|support|agent|fraud|pin|locked|failed|problem|reset|complain)"),
    ("VIEW_GROUP", r"\b(group|members?|contribut)"),
    ("MANAGE_CONTACTS", r"\b(contact|payee|recipient|save|saved|delete|edit|update)\b"),
    ("CHECK_SPENDING", r"\b(spend|spent|spending|expenditure|expense)"),
    ("VIEW_TRANSACTIONS", r"\b(transactions?|history|statement|activity|received|deposits?)\b"),
    ("WITHDRAW_MONEY", r"\b(withdraw|cash ?out|take out)"),
    ("BILL_PAY", r"\b(bill|invoice|fees?|electricity|water|efl|vodafone|digicel|internet|utility)"),
    ("CHECK_BALANCE", r"\b(balance|how much|funds|left|available)"),
    ("TRANSFER_MONEY", r"\b(send|transfer|pay)\b"),
]


def rule_predict(texts):
    out = []
    for t in texts:
        t = t.lower()
        out.append(next((lab for lab, pat in RULES if re.search(pat, t)), UNKNOWN))
    return np.array(out)


# --------------------------------------------------------------------------- helpers
def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def add_typos(text, rng):
    """One realistic keyboard slip in one word longer than 3 characters."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) > 3]
    if not idx:
        return text
    i = rng.choice(idx)
    w = list(words[i])
    p = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "dup", "sub"])
    if op == "swap":
        w[p], w[p + 1] = w[p + 1], w[p]
    elif op == "drop":
        del w[p]
    elif op == "dup":
        w.insert(p, w[p])
    else:
        w[p] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    words[i] = "".join(w)
    return " ".join(words)


def gate_metrics(proba, classes, y_true, thr, unknown_proba):
    conf = proba.max(1)
    pred = classes[proba.argmax(1)]
    acc_mask = conf >= thr
    wrong_accepted = acc_mask & (pred != y_true)
    return {
        "threshold": float(thr),
        "coverage_in_scope": float(acc_mask.mean()),
        "accuracy_when_accepted": float((pred[acc_mask] == y_true[acc_mask]).mean()),
        "wrong_workflow_rate": float(wrong_accepted.mean()),
        "wrong_money_workflow_rate": float((wrong_accepted & np.isin(pred, list(MONEY_MOVING))).mean()),
        "unknown_false_accept_rate": float((unknown_proba.max(1) >= thr).mean()),
    }


# --------------------------------------------------------------------------- main
def main():
    np.random.seed(SEED)
    df, P = load_data()
    tr, va, te, unk = P["train"], P["validation"], P["test"], P["evaluation_only"]
    assert set(unk.intent) == {UNKNOWN} and UNKNOWN not in set(tr.intent)
    print(f"train={len(tr)} validation={len(va)} test={len(te)} unknown={len(unk)}; "
          f"{tr.intent.nunique()} in-scope intents")

    # 1) hyperparameter search on the validation split
    search_rows, best = [], {}
    for name, grid in SEARCH_SPACE.items():
        scored = []
        for params in grid:
            m = build(name, params).fit(tr.text, tr.intent)
            f1 = f1_score(va.intent, m.predict(va.text), average="macro")
            scored.append((f1, params))
            search_rows.append({"model": name, "params": str(params), "val_macro_f1": round(f1, 4)})
        # best F1; ties broken toward the simpler (smaller C) model
        best_f1 = max(s[0] for s in scored)
        best[name] = next(p for f, p in scored if f >= best_f1 - 1e-9)
        print(f"{name:28s} best val macro-F1={best_f1:.4f} params={best[name]}")
    pd.DataFrame(search_rows).to_csv(OUT / "hyperparameter_search.csv", index=False)

    # 2) refit on train, freeze, evaluate once on the held-out test set
    rng = random.Random(SEED)
    te_typo = [add_typos(t, rng) for t in te.text]
    pd.DataFrame({"original": te.text, "with_typo": te_typo, "intent": te.intent}).to_csv(
        OUT / "test_with_typos.csv", index=False)

    summary, models, reports = [], {}, {}
    y_te = te.intent.values

    # business-rule benchmark (no training)
    rp = rule_predict(te.text)
    summary.append({
        "model": "Keyword rules (benchmark)",
        "val_macro_f1": round(f1_score(va.intent, rule_predict(va.text), average="macro",
                                       labels=sorted(set(y_te))), 4),
        "test_accuracy": round(accuracy_score(y_te, rp), 4),
        "test_macro_f1": round(f1_score(y_te, rp, average="macro", labels=sorted(set(y_te))), 4),
        "test_macro_f1_typos": round(f1_score(y_te, rule_predict(te_typo), average="macro",
                                              labels=sorted(set(y_te))), 4),
        "coverage_in_scope": round(float((rp != UNKNOWN).mean()), 4),
        "wrong_workflow_rate": round(float(((rp != y_te) & (rp != UNKNOWN)).mean()), 4),
        "unknown_false_accept_rate": round(float((rule_predict(unk.text) != UNKNOWN).mean()), 4),
    })

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    trva = pd.concat([tr, va])
    for name, params in best.items():
        m = build(name, params).fit(tr.text, tr.intent)
        models[name] = m
        classes = m.classes_
        pv, pt, pu = m.predict_proba(va.text), m.predict_proba(te.text), m.predict_proba(unk.text)
        # threshold chosen on VALIDATION only: keep TARGET_COVERAGE of valid commands
        thr = float(np.quantile(pv.max(1), 1 - TARGET_COVERAGE))
        g = gate_metrics(pt, classes, y_te, thr, pu)
        pred = m.predict(te.text)
        cv = cross_val_score(build(name, params), trva.text, trva.intent, cv=skf,
                             scoring="f1_macro")
        # latency (single command, CPU) and serialized size
        t0 = time.perf_counter()
        for t in te.text[:200]:
            m.predict_proba([t])
        latency_ms = (time.perf_counter() - t0) / 200 * 1000
        size_kb = len(pickle.dumps(m)) / 1024
        summary.append({
            "model": name,
            "params": str(params),
            "val_macro_f1": round(f1_score(va.intent, m.predict(va.text), average="macro"), 4),
            "cv5_macro_f1_mean": round(cv.mean(), 4),
            "cv5_macro_f1_std": round(cv.std(), 4),
            "test_accuracy": round(accuracy_score(y_te, pred), 4),
            "test_macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
            "test_macro_f1_typos": round(f1_score(y_te, m.predict(te_typo), average="macro"), 4),
            "test_errors": int((pred != y_te).sum()),
            **{k: round(v, 4) for k, v in g.items()},
            "latency_ms": round(latency_ms, 2),
            "model_size_kb": round(size_kb, 1),
        })
        reports[name] = classification_report(y_te, pred, output_dict=True, zero_division=0)
        # threshold sweep (diagnostic only; the operating point above is set on validation)
        sweep = [gate_metrics(pt, classes, y_te, t, pu) for t in np.linspace(0.1, 0.95, 35)]
        pd.DataFrame(sweep).to_csv(
            OUT / f"threshold_sweep_{slug(name)}.csv", index=False)
        models[name + "_sweep"] = sweep

    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "model_comparison.csv", index=False)

    # routing outcomes per 100,000 valid in-scope commands (input to the business case)
    impact = pd.DataFrame({
        "model": summ.model,
        "routed_to_correct_screen": (100_000 * (summ.coverage_in_scope - summ.wrong_workflow_rate)).round(-2),
        "sent_to_wrong_screen": (100_000 * summ.wrong_workflow_rate).round(-2),
        "sent_to_fallback": (100_000 * (1 - summ.coverage_in_scope)).round(-2),
        "out_of_scope_wrongly_accepted_pct": (100 * summ.unknown_false_accept_rate).round(1),
    })
    impact.to_csv(OUT / "routing_outcomes_per_100k.csv", index=False)
    print("\nRouting outcomes per 100k valid commands\n", impact.to_string(index=False))
    print("\n", summ.drop(columns=["params"], errors="ignore").to_string(index=False))

    # 3) error analysis for the best model
    best_name = max(best, key=lambda n: (summ.set_index("model").loc[n, "val_macro_f1"],
                                         summ.set_index("model").loc[n, "cv5_macro_f1_mean"]))
    bm = models[best_name]
    pred = bm.predict(te.text)
    conf = bm.predict_proba(te.text).max(1)
    errs = te.assign(predicted=pred, confidence=conf.round(3))[pred != y_te]
    errs.to_csv(OUT / "test_errors_best_model.csv", index=False)
    pu = bm.predict_proba(unk.text)
    unk.assign(predicted=bm.classes_[pu.argmax(1)], confidence=pu.max(1).round(3)) \
        .sort_values("confidence", ascending=False).head(25) \
        .to_csv(OUT / "unknown_highest_confidence_best_model.csv", index=False)

    per_class = pd.DataFrame({n: {c: reports[n][c]["f1-score"] for c in sorted(set(y_te))}
                              for n in best}).round(3)
    per_class.to_csv(OUT / "per_class_f1.csv")

    with open(OUT / "best_model.pkl", "wb") as f:
        pickle.dump(bm, f)
    json.dump({"best_model": best_name, "params": str(best[best_name]),
               "threshold": float(summ.set_index("model").loc[best_name, "threshold"]),
               "seed": SEED, "target_coverage": TARGET_COVERAGE},
              open(OUT / "best_model_config.json", "w"), indent=2)

    # top TF-IDF word features per intent (interpretability)
    wm = models["TF-IDF (word) + LR"]
    vocab = np.array(wm.named_steps["vec"].get_feature_names_out())
    coefs = wm.named_steps["clf"].coef_
    top = {c: ", ".join(vocab[np.argsort(coefs[i])[::-1][:8]]) for i, c in enumerate(wm.classes_)}
    pd.Series(top, name="top_terms").to_csv(OUT / "top_terms_tfidf_word.csv")

    make_figures(summ, models, best, best_name, te, y_te, per_class)
    out_of_scope_experiment(best[best_name], best_name, tr, va, te, unk)
    print(f"\nBest model: {best_name}. Outputs in {OUT}")


def out_of_scope_experiment(params, name, tr, va, te, unk):
    """EXPLORATORY (not part of the frozen benchmark): the confidence gate alone lets
    many out-of-scope requests through. Does adding explicit UNKNOWN_INTENT training
    examples help? We use half of the evaluation-only UNKNOWN set (150) for training
    and report on the other, unseen half (150) plus the unchanged in-scope test set."""
    u = unk.sample(frac=1, random_state=SEED).reset_index(drop=True)
    u_tr, u_te = u.iloc[:150], u.iloc[150:]
    rows = []
    for label, train_df in [("Gate only (benchmark set-up)", tr),
                            ("Gate + 150 UNKNOWN training examples", pd.concat([tr, u_tr]))]:
        m = build(name, params).fit(train_df.text, train_df.intent)
        pv = m.predict_proba(va.text)
        thr = float(np.quantile(pv.max(1), 1 - TARGET_COVERAGE))

        def route(texts):
            p = m.predict_proba(texts)
            lab = m.classes_[p.argmax(1)]
            return np.where((p.max(1) >= thr) & (lab != UNKNOWN), lab, UNKNOWN)

        r_te, r_u = route(te.text), route(u_te.text)
        rows.append({
            "setup": label, "threshold": round(thr, 3),
            "in_scope_routed_correctly": round(float((r_te == te.intent.values).mean()), 4),
            "in_scope_sent_to_fallback": round(float((r_te == UNKNOWN).mean()), 4),
            "in_scope_wrong_screen": round(float(((r_te != UNKNOWN) &
                                                  (r_te != te.intent.values)).mean()), 4),
            "unseen_unknown_rejected": round(float((r_u == UNKNOWN).mean()), 4),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "exploratory_out_of_scope.csv", index=False)
    print("\nExploratory out-of-scope experiment\n", res.to_string(index=False))


# --------------------------------------------------------------------------- figures
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
PAL = {"Keyword rules (benchmark)": "#9ca3af", "BoW + LR": "#93c5fd",
       "TF-IDF (word) + LR": "#3b82f6", "TF-IDF (word + char) + LR": "#1e3a8a"}


def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def make_figures(summ, models, best, best_name, te, y_te, per_class):
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    # A. clean vs typo macro F1
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(summ))
    w = 0.38
    for j, col, lab, alpha in [(-1, "test_macro_f1", "Clean test set", 1.0),
                               (1, "test_macro_f1_typos", "Test set with one typo per command", 0.45)]:
        bars = ax.bar(x + j * w / 2, summ[col], w, color=[PAL[m] for m in summ.model],
                      alpha=alpha, label=lab, edgecolor="white")
        for b, v in zip(bars, summ[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                    fontsize=9, color=INK)
    ax.set_xticks(x, [m.replace(" + LR", "\n+ LR").replace(" (benchmark)", "\n(benchmark)")
                      for m in summ.model])
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Macro F1 (held-out test)", color=MUTED)
    ax.set_title("Richer text representation holds up when users make typos",
                 loc="left", color=INK, fontsize=13)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=200)
    plt.close(fig)

    # B. confusion matrix, best model
    m = models[best_name]
    labels = list(m.classes_)
    cm = confusion_matrix(y_te, m.predict(te.text), labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > 15 else INK, fontsize=10)
    ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), [l for l in labels], fontsize=8)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Actual", color=MUTED)
    ax.set_title(f"Confusion matrix - {best_name} (test, n=270)", loc="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix_best.png", dpi=200)
    plt.close(fig)

    # C. threshold trade-off, best model
    sw = pd.DataFrame(models[best_name + "_sweep"])
    thr = summ.set_index("model").loc[best_name, "threshold"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(sw.threshold, sw.coverage_in_scope, color="#1e3a8a", lw=2.2,
            label="Valid commands routed automatically")
    ax.plot(sw.threshold, sw.unknown_false_accept_rate, color="#dc2626", lw=2.2,
            label="Out-of-scope requests wrongly accepted")
    ax.plot(sw.threshold, sw.wrong_workflow_rate, color="#f59e0b", lw=2.2,
            label="Valid commands sent to the wrong screen")
    ax.axvline(thr, color=MUTED, ls="--", lw=1)
    ax.text(thr + 0.01, 0.12, f"operating point\n(set on validation)\nthreshold = {thr:.2f}",
            color=MUTED, fontsize=9)
    ax.set_xlabel("Confidence threshold", color=MUTED)
    ax.set_ylabel("Share of commands", color=MUTED)
    ax.set_ylim(0, 1.03)
    ax.set_title("Confidence gate: automation vs. safety", loc="left", color=INK, fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.5))
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "threshold_tradeoff_best.png", dpi=200)
    plt.close(fig)

    # D. per-class F1
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pc = per_class[list(best)]
    x = np.arange(len(pc))
    w = 0.27
    for k, n in enumerate(pc.columns):
        ax.bar(x + (k - 1) * w, pc[n], w, color=PAL[n], label=n, edgecolor="white")
    ax.set_xticks(x, [i.replace("_", "\n", 1) for i in pc.index], fontsize=8)
    ax.set_ylim(min(0.6, pc.values.min() - 0.05), 1.02)
    ax.set_ylabel("F1 (test)", color=MUTED)
    ax.set_title("Per-intent F1 on the held-out test set", loc="left", color=INK, fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "per_class_f1.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
