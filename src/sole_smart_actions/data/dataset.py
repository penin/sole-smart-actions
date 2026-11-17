"""Load the canonical V5 workbook, validate it, and expose the predefined splits.

Nothing here modifies, re-splits, re-labels or regenerates data. Validation raises
:class:`DatasetValidationError` on any failure so no model can train on unexpected data.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from sole_intent.preprocessing import dedup_key

from ..config import rel, resolve
from ..utils import sha256_file, sha256_text

SPLITS = ("train", "validation", "test", "evaluation_only")


class DatasetValidationError(RuntimeError):
    """The workbook does not match the canonical dataset; training must stop."""


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    unknown: pd.DataFrame

    def items(self):
        return {"train": self.train, "validation": self.validation, "test": self.test,
                "unknown": self.unknown}.items()


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Read the training sheet as text. ``row_id`` and ``excel_row`` trace predictions back to the workbook."""
    d = cfg["dataset"]
    path = resolve(d["path"])
    if not path.exists():
        raise DatasetValidationError(f"Dataset not found: {rel(path)}")
    df = pd.read_excel(path, sheet_name=d["sheet"], dtype=str, keep_default_na=False, na_values=[""],
                       engine="openpyxl")
    missing = [c for c in d["columns"] if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}; found {list(df.columns)}")
    df = df[d["columns"]].copy()
    df.insert(0, "row_id", range(len(df)))
    df.insert(1, "excel_row", df["row_id"] + 2)
    return df


def dataset_fingerprint(cfg: dict, df: pd.DataFrame | None = None) -> dict:
    path = resolve(cfg["dataset"]["path"])
    df = load_dataset(cfg) if df is None else df
    content = df[cfg["dataset"]["columns"]].to_csv(index=False, lineterminator="\n")
    return {"path": rel(path), "version": cfg["dataset"]["version"], "sha256": sha256_file(path),
            "content_sha256": sha256_text(content), "n_rows": int(len(df))}


def validate_dataset(cfg: dict, df: pd.DataFrame | None = None, *, raise_on_failure: bool = True) -> dict:
    """Run every check required by the shared protocol and return a report.

    Checks: file hash, schema, missing values, label/split vocabularies, split sizes, 300 examples per
    intent (including UNKNOWN_INTENT), per-split class balance, exact duplicates, normalized duplicates,
    conflicting labels, and overlap between train / validation / test / evaluation_only.
    """
    d = cfg["dataset"]
    labels, unknown = list(cfg["labels"]), cfg["unknown_label"]
    df = load_dataset(cfg) if df is None else df
    checks: list[dict] = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    fp = dataset_fingerprint(cfg, df)
    expected_sha = (d.get("sha256") or "").lower()
    check("file_sha256_matches_config", not expected_sha or fp["sha256"] == expected_sha,
          {"expected": expected_sha or None, "actual": fp["sha256"]})
    check("schema_columns", list(df.columns[2:]) == list(d["columns"]), list(df.columns[2:]))
    na = df[d["columns"]].isna().sum()
    blank = (df["text"].fillna("").str.strip() == "").sum()
    check("no_missing_values", int(na.sum()) == 0 and int(blank) == 0,
          {"missing_by_column": {k: int(v) for k, v in na.items()}, "blank_text": int(blank)})
    check("row_count", len(df) == d["expected_rows"], {"expected": d["expected_rows"], "actual": len(df)})

    bad_intents = sorted(set(df["intent"].dropna()) - set(labels) - {unknown})
    check("intent_vocabulary", not bad_intents, {"unexpected": bad_intents})
    bad_splits = sorted(set(df["split"].dropna()) - set(SPLITS))
    check("split_vocabulary", not bad_splits, {"unexpected": bad_splits})

    counts = df["split"].value_counts().to_dict()
    check("split_sizes", all(counts.get(s, 0) == n for s, n in d["expected_split_counts"].items()),
          {"expected": d["expected_split_counts"], "actual": {s: int(counts.get(s, 0)) for s in SPLITS}})

    per_intent = df["intent"].value_counts().to_dict()
    want = d["per_intent"]
    check("examples_per_intent", all(per_intent.get(c, 0) == want for c in labels + [unknown]),
          {"expected_each": want, "actual": {c: int(per_intent.get(c, 0)) for c in labels + [unknown]}})

    unknown_rows = df["intent"] == unknown
    check("unknown_only_in_evaluation_only",
          bool((df.loc[unknown_rows, "split"] == "evaluation_only").all())
          and bool((df.loc[df["split"] == "evaluation_only", "intent"] == unknown).all()),
          {"unknown_outside_evaluation_only": int((unknown_rows & (df["split"] != "evaluation_only")).sum()),
           "supported_in_evaluation_only": int((~unknown_rows & (df["split"] == "evaluation_only")).sum())})

    dist = pd.crosstab(df.loc[~unknown_rows, "intent"], df.loc[~unknown_rows, "split"])
    n_cls = len(labels)
    balanced = all(dist.get(s, pd.Series(dtype=int)).reindex(labels, fill_value=0).eq(
        d["expected_split_counts"][s] // n_cls).all() for s in ("train", "validation", "test"))
    check("class_distribution_per_split", balanced,
          {s: {c: int(dist.at[c, s]) if c in dist.index and s in dist.columns else 0 for c in labels}
           for s in ("train", "validation", "test")})

    exact_dupes = df[df.duplicated("text", keep=False)]
    check("no_exact_duplicates", exact_dupes.empty,
          {"n_rows": int(len(exact_dupes)), "examples": exact_dupes["text"].head(5).tolist()})
    keys = df["text"].map(dedup_key)
    norm_dupes = df[keys.duplicated(keep=False)]
    check("no_normalized_duplicates", norm_dupes.empty,
          {"normalization": "NFKC + casefold + collapsed whitespace + stripped surrounding punctuation",
           "n_rows": int(len(norm_dupes)), "examples": norm_dupes["text"].head(5).tolist()})
    conflicts = df.assign(k=keys).groupby("k")["intent"].nunique()
    conflicts = conflicts[conflicts > 1]
    check("no_conflicting_labels", conflicts.empty,
          {"n_texts": int(len(conflicts)), "examples": conflicts.index[:5].tolist()})

    split_keys = {s: set(keys[df["split"] == s]) for s in SPLITS}
    overlap = {f"{a}&{b}": len(split_keys[a] & split_keys[b]) for a, b in combinations(SPLITS, 2)}
    check("no_cross_split_overlap", all(v == 0 for v in overlap.values()), overlap)

    report = {"dataset": fp, "passed": all(c["passed"] for c in checks), "checks": checks,
              "n_checks": len(checks), "n_failed": sum(not c["passed"] for c in checks)}
    if raise_on_failure and not report["passed"]:
        failed = [c["check"] for c in checks if not c["passed"]]
        raise DatasetValidationError(f"Dataset validation failed ({rel(resolve(d['path']))}): {failed}")
    return report


def load_splits(cfg: dict, *, validate: bool = True) -> tuple[Splits, dict]:
    """Validated fixed splits plus the dataset fingerprint. Supervised splits never contain UNKNOWN_INTENT."""
    df = load_dataset(cfg)
    report = validate_dataset(cfg, df) if validate else {"dataset": dataset_fingerprint(cfg, df)}
    unknown = cfg["unknown_label"]
    sup = df[df["intent"] != unknown]

    def pick(frame, split):
        return frame[frame["split"] == split].reset_index(drop=True)

    splits = Splits(train=pick(sup, "train"), validation=pick(sup, "validation"), test=pick(sup, "test"),
                    unknown=pick(df[df["intent"] == unknown], "evaluation_only"))
    assert unknown not in set(splits.train["intent"]), "UNKNOWN_INTENT must never be a training class"
    return splits, report["dataset"]
