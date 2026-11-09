import copy

import pandas as pd
import pytest

from sole_intent.audit import audit_dataset
from sole_intent.data_loader import (REQUIRED_COLUMNS, DatasetError, dataset_path, get_splits, load_dataset,
                                     validate_schema)
from sole_intent.utils import sha256_file


def test_dataset_loads_with_expected_schema(df, cfg):
    assert len(df) == cfg["data"]["expected_rows"] == 3000
    assert list(df.columns) == ["row_id", "excel_row", *REQUIRED_COLUMNS]
    assert df[REQUIRED_COLUMNS].notna().all().all()
    assert (df["text"].str.strip() != "").all()


def test_supported_labels(df, cfg):
    assert len(cfg["labels"]) == 9
    assert set(df["intent"]) == set(cfg["labels"]) | {cfg["unknown_label"]}
    assert set(cfg["core_labels"]) == set(cfg["labels"]) - {"CHECK_SPENDING"}
    validate_schema(df, cfg)  # must not raise


def test_split_assignments(df, cfg):
    assert df["split"].value_counts().to_dict() == cfg["data"]["expected_split_counts"]
    supported = df[df["intent"] != cfg["unknown_label"]]
    counts = pd.crosstab(supported["intent"], supported["split"])
    assert (counts["train"] == 240).all() and (counts["validation"] == 30).all() and (counts["test"] == 30).all()


def test_scope_column_preserved(df, cfg):
    assert set(df.loc[df["intent"] == "CHECK_SPENDING", "scope"]) == {"stretch"}
    assert set(df.loc[df["intent"].isin(cfg["core_labels"]), "scope"]) == {"core"}
    assert set(df.loc[df["intent"] == cfg["unknown_label"], "scope"]) == {"rejection_test"}


def test_no_unknown_rows_in_classifier_splits(splits, cfg):
    for name in ("train", "validation", "test"):
        assert cfg["unknown_label"] not in set(getattr(splits, name)["intent"])
    assert len(splits.train) == 2160
    assert set(splits.unknown["intent"]) == {cfg["unknown_label"]}
    assert set(splits.unknown["split"]) == {"evaluation_only"}
    assert len(splits.unknown) == 300


def test_splits_are_disjoint_and_complete(splits, df):
    ids = [set(f["row_id"]) for f in splits.as_dict().values()]
    assert sum(len(s) for s in ids) == len(df)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not ids[i] & ids[j]


def test_loading_does_not_modify_source(cfg):
    before = sha256_file(dataset_path(cfg))
    load_dataset(cfg)
    assert sha256_file(dataset_path(cfg)) == before


def test_missing_dataset_raises(cfg):
    bad = copy.deepcopy(cfg)
    bad["data"]["path"] = "does_not_exist.xlsx"
    with pytest.raises(DatasetError):
        load_dataset(bad)


def test_validate_schema_rejects_bad_rows(df, cfg):
    broken = df.copy()
    broken.loc[0, "intent"] = "ORDER_PIZZA"
    with pytest.raises(DatasetError, match="invalid intents"):
        validate_schema(broken, cfg)
    leaked = df.copy()
    idx = leaked.index[leaked["intent"] == cfg["unknown_label"]][0]
    leaked.loc[idx, "split"] = "train"
    with pytest.raises(DatasetError, match="UNKNOWN_INTENT"):
        validate_schema(leaked, cfg)


def test_get_splits_never_trains_on_unknown_even_if_mislabelled(df, cfg):
    tampered = df.copy()
    idx = tampered.index[tampered["intent"] == cfg["unknown_label"]][:5]
    tampered.loc[idx, "split"] = "train"
    assert cfg["unknown_label"] not in set(get_splits(tampered, cfg).train["intent"])


def test_audit_reports_without_mutating(df, cfg):
    snapshot = df.copy()
    summary, findings, details = audit_dataset(df, cfg)
    pd.testing.assert_frame_equal(df, snapshot)
    assert summary["contradictory_label_phrases"] == 0
    assert not (findings.query("category == 'schema'")["status"] == "fail").any()
    assert {"duplicate_groups", "cross_split_overlap", "leakage_rows", "length_stats"} <= set(details)
    # leakage audit should find the known train/test phrase overlap rather than hide it
    assert summary["leakage"]["test"]["normalized_in_train"] > 0
