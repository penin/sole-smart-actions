"""Dataset quality audit: schema, balance, lengths, duplicates, label conflicts and leakage.

The audit only *reports*. It never deletes or edits rows; leakage-controlled evaluation
subsets are derived separately and the primary benchmark keeps the supplied splits.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .data_loader import REQUIRED_COLUMNS, VALID_SPLITS
from .preprocessing import canonical_key, template_key

SPLIT_ORDER = ["train", "validation", "test", "evaluation_only"]


def add_audit_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["canonical"] = out["text"].map(canonical_key)
    out["template"] = out["text"].map(template_key)
    out["n_tokens"] = out["canonical"].map(lambda s: len(s.split()) if s else 0)
    out["n_chars"] = out["text"].fillna("").str.len()
    return out


def _jaccard_max(query_sets: list[set], reference_sets: list[set]) -> np.ndarray:
    best = np.zeros(len(query_sets))
    for i, q in enumerate(query_sets):
        if not q:
            continue
        best[i] = max((len(q & r) / len(q | r) for r in reference_sets if r), default=0.0)
    return best


def leakage_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-row leakage flags for every non-train row relative to the training split."""
    keyed = add_audit_keys(df)
    train = keyed[keyed["split"] == "train"]
    train_canon, train_tmpl = set(train["canonical"]), set(train["template"])
    train_sets = [set(s.split()) for s in train["canonical"]]
    others = keyed[keyed["split"] != "train"].copy()
    others["exact_in_train"] = others["text"].isin(set(train["text"]))
    others["canonical_in_train"] = others["canonical"].isin(train_canon)
    others["template_in_train"] = others["template"].isin(train_tmpl)
    others["max_jaccard_to_train"] = _jaccard_max([set(s.split()) for s in others["canonical"]], train_sets)
    thr = cfg["diagnostics"]["near_duplicate_jaccard"]
    others["near_duplicate_of_train"] = others["max_jaccard_to_train"] >= thr
    others["leakage_flag"] = others["canonical_in_train"] | others["template_in_train"]
    return others


def audit_dataset(df: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (summary dict, tidy findings table, detail tables)."""
    labels, unknown = cfg["labels"], cfg["unknown_label"]
    scopes = cfg["scopes"]
    keyed = add_audit_keys(df)
    rows: list[dict] = []

    def finding(category, check, value, status, details=""):
        rows.append({"category": category, "check": check, "value": value,
                     "status": status, "details": details})

    # --- Schema and integrity -------------------------------------------------------
    n = len(df)
    finding("schema", "row_count", n, "ok" if n == cfg["data"]["expected_rows"] else "warn",
            f"expected {cfg['data']['expected_rows']}")
    missing = df[REQUIRED_COLUMNS].isna().sum().to_dict()
    for col, cnt in missing.items():
        finding("schema", f"missing_{col}", int(cnt), "ok" if cnt == 0 else "fail")
    empty_text = int(df["text"].fillna("").str.strip().eq("").sum())
    finding("schema", "empty_or_whitespace_text", empty_text, "ok" if empty_text == 0 else "fail")
    invalid_intents = sorted(set(df["intent"].dropna()) - set(labels) - {unknown})
    finding("schema", "invalid_intent_labels", len(invalid_intents),
            "ok" if not invalid_intents else "fail", ", ".join(invalid_intents))
    invalid_splits = sorted(set(df["split"].dropna()) - set(VALID_SPLITS))
    finding("schema", "invalid_split_values", len(invalid_splits),
            "ok" if not invalid_splits else "fail", ", ".join(invalid_splits))

    expected_scope = {lab: scopes["core"] for lab in cfg["core_labels"]}
    expected_scope.update({lab: scopes["stretch"] for lab in labels if lab not in cfg["core_labels"]})
    expected_scope[unknown] = scopes["rejection"]
    scope_mismatch = int((df["scope"] != df["intent"].map(expected_scope)).sum())
    finding("schema", "scope_intent_mismatch", scope_mismatch, "ok" if scope_mismatch == 0 else "warn",
            "core -> 8 core intents; stretch -> CHECK_SPENDING; rejection_test -> UNKNOWN_INTENT")
    unk_outside = int(((df["intent"] == unknown) & (df["split"] != "evaluation_only")).sum())
    finding("schema", "unknown_rows_outside_evaluation_only", unk_outside, "ok" if unk_outside == 0 else "fail")

    # --- Split sizes and balance ----------------------------------------------------
    split_counts = df["split"].value_counts().to_dict()
    for split, expected in cfg["data"]["expected_split_counts"].items():
        got = int(split_counts.get(split, 0))
        finding("splits", f"count_{split}", got, "ok" if got == expected else "warn", f"expected {expected}")
    balance = pd.crosstab(df["intent"], df["split"]).reindex(columns=SPLIT_ORDER, fill_value=0)
    for split in ["train", "validation", "test"]:
        col = balance.loc[labels, split]
        ratio = float(col.max() / col.min()) if col.min() > 0 else float("inf")
        finding("balance", f"imbalance_ratio_{split}", round(ratio, 3),
                "ok" if ratio <= 1.5 else "warn", f"min={int(col.min())} max={int(col.max())}")

    # --- Duplicates -----------------------------------------------------------------
    exact_dup_rows = int(df["text"].duplicated(keep=False).sum())
    canon_counts = keyed["canonical"].value_counts()
    canon_dup_rows = int(keyed["canonical"].isin(canon_counts[canon_counts > 1].index).sum())
    tmpl_counts = keyed["template"].value_counts()
    tmpl_dup_rows = int(keyed["template"].isin(tmpl_counts[tmpl_counts > 1].index).sum())
    finding("duplicates", "exact_duplicate_rows", exact_dup_rows, "warn" if exact_dup_rows else "ok",
            f"{int(df['text'].duplicated().sum())} redundant copies")
    finding("duplicates", "normalized_duplicate_rows", canon_dup_rows, "warn" if canon_dup_rows else "ok",
            f"{int(keyed['canonical'].duplicated().sum())} redundant copies; "
            f"{int((canon_counts > 1).sum())} repeated phrases")
    finding("duplicates", "number_masked_duplicate_rows", tmpl_dup_rows, "info",
            "same phrase after masking numbers (e.g. 'send 50' vs 'send 500')")
    for split in SPLIT_ORDER:
        sub = keyed[keyed["split"] == split]
        redundant = int(sub["canonical"].duplicated().sum())
        finding("duplicates", f"within_split_redundant_{split}", redundant, "warn" if redundant else "ok",
                f"{sub['canonical'].nunique()} unique of {len(sub)}")

    # Cross-partition overlap (any pair of partitions)
    cross_rows = []
    split_sets = {s: set(keyed.loc[keyed["split"] == s, "canonical"]) for s in SPLIT_ORDER}
    split_tmpl = {s: set(keyed.loc[keyed["split"] == s, "template"]) for s in SPLIT_ORDER}
    for a, b in combinations(SPLIT_ORDER, 2):
        shared = split_sets[a] & split_sets[b]
        shared_t = split_tmpl[a] & split_tmpl[b]
        rows_b = int(keyed[(keyed["split"] == b) & keyed["canonical"].isin(shared)].shape[0])
        status = "ok" if not shared else ("fail" if a == "train" and b == "test" else "warn")
        finding("leakage", f"shared_phrases_{a}_vs_{b}", len(shared), status,
                f"{rows_b} {b} rows; {len(shared_t)} shared number-masked templates")
        for phrase in sorted(shared | shared_t):
            cross_rows.append({"partition_a": a, "partition_b": b, "phrase_or_template": phrase,
                               "match_type": "normalized" if phrase in shared else "number_masked"})

    # Contradictory labels
    intents_per_phrase = keyed.groupby("canonical")["intent"].nunique()
    conflicts = intents_per_phrase[intents_per_phrase > 1]
    finding("labels", "contradictory_label_phrases", int(len(conflicts)),
            "ok" if conflicts.empty else "fail", "same normalized phrase, different intent")
    tmpl_conf = keyed.groupby("template")["intent"].nunique()
    finding("labels", "contradictory_label_templates", int((tmpl_conf > 1).sum()),
            "ok" if (tmpl_conf <= 1).all() else "warn", "same number-masked template, different intent")

    # Near-duplicates / leakage relative to train
    leak = leakage_table(df, cfg)
    for split in ["validation", "test", "evaluation_only"]:
        sub = leak[leak["split"] == split]
        finding("leakage", f"{split}_rows_normalized_in_train", int(sub["canonical_in_train"].sum()),
                "ok" if not sub["canonical_in_train"].any() else ("fail" if split == "test" else "warn"))
        finding("leakage", f"{split}_rows_template_in_train", int(sub["template_in_train"].sum()),
                "ok" if not sub["template_in_train"].any() else "warn")
        finding("leakage", f"{split}_rows_near_duplicate_of_train", int(sub["near_duplicate_of_train"].sum()),
                "info", f"token Jaccard >= {cfg['diagnostics']['near_duplicate_jaccard']}; "
                        f"median max-Jaccard {sub['max_jaccard_to_train'].median():.2f}")

    # --- Length distribution --------------------------------------------------------
    length_stats = (keyed.groupby(["split", "intent"])[["n_tokens", "n_chars"]]
                    .agg(["mean", "median", "min", "max"]).round(2))
    length_stats.columns = ["_".join(c) for c in length_stats.columns]
    length_stats = length_stats.reset_index()
    overall_tok = keyed["n_tokens"].describe()
    finding("length", "tokens_mean", round(float(overall_tok["mean"]), 2), "info",
            f"min={int(overall_tok['min'])} median={overall_tok['50%']:.0f} max={int(overall_tok['max'])}")
    one_token = int((keyed["n_tokens"] <= 1).sum())
    finding("length", "single_token_utterances", one_token, "info")

    findings = pd.DataFrame(rows)
    duplicate_groups = (keyed.groupby("canonical")
                        .agg(count=("text", "size"),
                             splits=("split", lambda s: "|".join(sorted(set(s)))),
                             intents=("intent", lambda s: "|".join(sorted(set(s)))),
                             examples=("text", lambda s: " || ".join(list(dict.fromkeys(s))[:3])))
                        .query("count > 1").sort_values("count", ascending=False).reset_index())
    leak_cols = ["row_id", "excel_row", "text", "intent", "split", "canonical", "template",
                 "canonical_in_train", "template_in_train", "max_jaccard_to_train",
                 "near_duplicate_of_train", "leakage_flag"]
    summary = {
        "n_rows": n,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "class_balance": balance.to_dict(orient="index"),
        "scope_counts": df["scope"].value_counts().to_dict(),
        "missing_values": missing,
        "empty_text": empty_text,
        "invalid_intents": invalid_intents,
        "exact_duplicate_redundant_rows": int(df["text"].duplicated().sum()),
        "normalized_duplicate_redundant_rows": int(keyed["canonical"].duplicated().sum()),
        "unique_normalized_phrases": int(keyed["canonical"].nunique()),
        "unique_phrases_per_split": {s: int(keyed.loc[keyed["split"] == s, "canonical"].nunique())
                                     for s in SPLIT_ORDER},
        "contradictory_label_phrases": int(len(conflicts)),
        "leakage": {
            split: {
                "rows": int((leak["split"] == split).sum()),
                "normalized_in_train": int(leak.loc[leak["split"] == split, "canonical_in_train"].sum()),
                "template_in_train": int(leak.loc[leak["split"] == split, "template_in_train"].sum()),
                "near_duplicate_of_train": int(leak.loc[leak["split"] == split, "near_duplicate_of_train"].sum()),
                "leakage_flagged": int(leak.loc[leak["split"] == split, "leakage_flag"].sum()),
            } for split in ["validation", "test", "evaluation_only"]
        },
        "token_length": {k: float(v) for k, v in overall_tok.items()},
        "status_counts": findings["status"].value_counts().to_dict(),
    }
    details = {
        "duplicate_groups": duplicate_groups,
        "cross_split_overlap": pd.DataFrame(cross_rows),
        "leakage_rows": leak[leak_cols].sort_values(["split", "row_id"]).reset_index(drop=True),
        "length_stats": length_stats,
    }
    return summary, findings, details
