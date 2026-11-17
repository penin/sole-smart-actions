"""Canonical Dataset V5: loading, validation and split integrity (sole_smart_actions.data)."""

import pandas as pd
import pytest

from sole_smart_actions import MODEL_NAMES
from sole_smart_actions.config import load_config
from sole_smart_actions.data import DatasetValidationError, load_dataset, load_splits, validate_dataset


@pytest.fixture(scope="module")
def common():
    return load_config()


@pytest.fixture(scope="module")
def df(common):
    return load_dataset(common)


def failed(report):
    return {c["check"] for c in report["checks"] if not c["passed"]}


def test_canonical_dataset_passes_every_check(common, df):
    report = validate_dataset(common, df)
    assert report["passed"] and report["n_checks"] >= 14 and not failed(report)
    assert report["dataset"]["sha256"] == common["dataset"]["sha256"]


def test_split_sizes_and_class_balance(common):
    splits, fp = load_splits(common)
    assert {k: len(v) for k, v in splits.items()} == {"train": 2160, "validation": 270, "test": 270, "unknown": 300}
    for name in ("train", "validation", "test"):
        counts = getattr(splits, name)["intent"].value_counts()
        assert sorted(counts.index) == sorted(common["labels"]) and counts.nunique() == 1
    assert set(splits.unknown["intent"]) == {"UNKNOWN_INTENT"}
    assert "UNKNOWN_INTENT" not in set(splits.train["intent"]) | set(splits.validation["intent"]) | set(splits.test["intent"])


def test_splits_are_disjoint(common):
    splits, _ = load_splits(common)
    keys = {k: set(v["text"].str.lower().str.strip()) for k, v in splits.items()}
    for a in keys:
        for b in keys:
            if a < b:
                assert not keys[a] & keys[b], (a, b)


@pytest.mark.parametrize("mutation, check", [
    (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "no_exact_duplicates"),
    (lambda d: d.assign(text=d["text"].where(d.index != 1, d.loc[0, "text"].upper() + "!")), "no_normalized_duplicates"),
    (lambda d: d.assign(text=d["text"].where(d.index != 1, d.loc[0, "text"]),
                        intent=d["intent"].where(d.index != 1, "BILL_PAY" if d.loc[0, "intent"] != "BILL_PAY" else "VIEW_GROUP")),
     "no_conflicting_labels"),
    (lambda d: d.assign(text=d["text"].where(d.index != 5, None)), "no_missing_values"),
    (lambda d: d.assign(split=d["split"].where(d.index != 0, "test")), "split_sizes"),
    (lambda d: d.assign(intent=d["intent"].where(d.index != 0, "NOT_AN_INTENT")), "intent_vocabulary"),
])
def test_validation_detects_corruption(common, df, mutation, check):
    bad = mutation(df.copy())
    report = validate_dataset(common, bad, raise_on_failure=False)
    assert check in failed(report)
    with pytest.raises(DatasetValidationError):
        validate_dataset(common, bad)


def test_cross_split_overlap_detected(common, df):
    bad = df.copy()
    test_row = bad.index[bad["split"] == "test"][0]
    train_row = bad.index[bad["split"] == "train"][0]
    bad.loc[test_row, ["text", "intent"]] = bad.loc[train_row, ["text", "intent"]].to_numpy()
    assert {"no_cross_split_overlap", "no_exact_duplicates"} <= failed(validate_dataset(common, bad, raise_on_failure=False))


def test_wrong_file_hash_stops_training(common, df):
    cfg = {**common, "dataset": {**common["dataset"], "sha256": "0" * 64}}
    assert "file_sha256_matches_config" in failed(validate_dataset(cfg, df, raise_on_failure=False))


def test_every_model_uses_the_same_dataset_and_class_order(common):
    for name in MODEL_NAMES:
        cfg = load_config(name)
        assert cfg["dataset"] == common["dataset"] and cfg["labels"] == common["labels"] and len(cfg["labels"]) == 9
        assert cfg["unknown_label"] not in cfg["labels"]
