"""Dataset-versioned runs (V5): output isolation, pre-training validation and comparison helpers."""

import copy
from pathlib import Path

import pytest

from sole_intent.audit import strict_dataset_checks
from sole_intent.compare import date_template_key
from sole_intent.data_loader import load_dataset
from sole_intent.experiment import guard_output_dataset
from sole_intent.preprocessing import dedup_key
from sole_intent.utils import PROJECT_ROOT, load_config, output_paths, save_json

V5 = PROJECT_ROOT / "Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx"
needs_v5 = pytest.mark.skipif(not V5.exists(), reason="Dataset V5 workbook not present")


def test_default_output_paths_are_the_original_v4_locations(cfg):
    P = output_paths(cfg)
    assert P.results == PROJECT_ROOT / "results" and P.models == PROJECT_ROOT / "models"
    assert P.processed == PROJECT_ROOT / "data" / "processed"
    assert P.metrics == PROJECT_ROOT / "results" / "metrics"


def test_v5_config_isolated_and_methodology_identical():
    v4, v5 = load_config("config/baseline.yaml"), load_config("config/baseline_v5.yaml")
    p4, p5 = output_paths(v4), output_paths(v5)
    assert {p4.results, p4.models, p4.processed}.isdisjoint({p5.results, p5.models, p5.processed})
    allowed = {"experiment", "data", "outputs", "validation", "threshold_sweep", "comparison"}
    for key in set(v4) | set(v5):
        if key not in allowed:
            assert v4.get(key) == v5.get(key), key
    assert v4["experiment"]["random_seed"] == v5["experiment"]["random_seed"]
    assert {k: v for k, v in v4["data"].items() if k not in ("path", "dataset_version")} == \
           {k: v for k, v in v5["data"].items() if k not in ("path", "dataset_version")}


def test_guard_refuses_to_overwrite_results_of_another_dataset(tmp_path):
    P = output_paths({"outputs": {"results_dir": str(tmp_path / "r"), "models_dir": str(tmp_path / "m"),
                                  "processed_dir": str(tmp_path / "p")}})
    guard_output_dataset(P, {"content_sha256": "aaa"})  # empty folder: fine
    save_json({"dataset": {"content_sha256": "aaa"}}, P.metrics / "experiment_config.json")
    guard_output_dataset(P, {"content_sha256": "aaa"})  # same dataset: fine (idempotent rerun)
    with pytest.raises(RuntimeError, match="different dataset"):
        guard_output_dataset(P, {"content_sha256": "bbb"})


def test_v4_results_folder_is_protected_from_v5_data():
    v5 = load_config("config/baseline_v5.yaml")
    v4_paths = output_paths(load_config("config/baseline.yaml"))
    if not (v4_paths.metrics / "experiment_config.json").exists():
        pytest.skip("V4 results not present")
    with pytest.raises(RuntimeError):
        guard_output_dataset(v4_paths, {"content_sha256": "v5-would-not-match"})
    assert output_paths(v5).results != v4_paths.results


@pytest.mark.parametrize("a, b", [("Send $50 to Jone", "  send   $50 to jone?? "), ("ＳＥＮＤ", "send"),
                                  ("“Pay my bill.”", "pay my bill")])
def test_dedup_key_normalizes_case_space_unicode_surrounding_punct(a, b):
    assert dedup_key(a) == dedup_key(b)


def test_dedup_key_keeps_interior_punctuation():
    assert dedup_key("App keeps crashing? asap") != dedup_key("App keeps crashing asap")
    assert dedup_key(None) == ""


def test_date_template_key_masks_numbers_and_months():
    assert date_template_key("Show money received on 8 February") == date_template_key("show money received on 20 March")
    assert date_template_key("send 50 to jone") != date_template_key("send 50 to mere")


@needs_v5
def test_strict_checks_pass_on_v5():
    cfg = load_config("config/baseline_v5.yaml")
    result = strict_dataset_checks(load_dataset(cfg), cfg)
    assert result["passed"], result["failed_checks"]
    names = {c["check"] for c in result["checks"] if c["status"] == "pass"}
    assert {"total_rows", "rows_train", "rows_validation", "rows_test", "rows_evaluation_only",
            "rows_per_intent_including_unknown", "exact_duplicate_commands", "normalized_duplicate_commands",
            "cross_split_duplicate_commands", "conflicting_labels_same_normalized_command",
            "unknown_intent_evaluation_only", "missing_command_text", "missing_intent_label"} <= names


def test_strict_checks_catch_v4_duplicates(cfg):
    c = copy.deepcopy(cfg)
    c["validation"] = {"per_intent": 300}
    result = strict_dataset_checks(load_dataset(c), c)
    assert not result["passed"]
    assert {"exact_duplicate_commands", "normalized_duplicate_commands"} <= set(result["failed_checks"])


@needs_v5
def test_strict_checks_fail_on_injected_problems():
    cfg = load_config("config/baseline_v5.yaml")
    df = load_dataset(cfg)
    bad = df.copy()
    bad.loc[1, "text"] = bad.loc[0, "text"].upper() + "!"          # normalized duplicate across splits
    bad.loc[2, "intent"] = "UNKNOWN_INTENT"                          # unknown in train + count drift
    failed = set(strict_dataset_checks(bad, cfg)["failed_checks"])
    assert {"normalized_duplicate_commands", "cross_split_duplicate_commands",
            "unknown_intent_evaluation_only", "rows_per_intent_including_unknown"} <= failed
    assert df.equals(load_dataset(cfg))  # checks never modify the data
