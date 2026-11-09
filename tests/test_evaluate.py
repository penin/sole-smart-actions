"""Metric correctness on small hand-computable fixtures."""

import math

import numpy as np
import pytest

from sole_intent.evaluate import (SUMMARY_FIELDS, cluster_bootstrap, core_breakdown, evaluate_predictions,
                                  expected_calibration_error, measure_latency, multiclass_log_loss, summary_row)

CLASSES = ["A", "B", "C"]


def _eval(y_true, y_pred, proba, classes=CLASSES):
    return evaluate_predictions(y_true, y_pred, proba, classes, "unit", "toy", "exp0")


def test_perfect_predictions():
    y = ["A", "B", "C", "A"]
    proba = np.eye(3)[[0, 1, 2, 0]] * 0.97 + 0.01
    m = _eval(y, y, proba)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["weighted_f1"] == 1
    assert m["macro_roc_auc_ovr"] == 1 and m["balanced_accuracy"] == 1
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_hand_computed_metrics():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    proba = np.array([[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
    m = evaluate_predictions(y_true, y_pred, proba, ["A", "B", "C"], "unit", "toy", "exp0")
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["top2_accuracy"] == 1.0
    pc = {r["intent"]: r for r in m["per_class"]}
    assert pc["A"]["precision"] == 1.0 and pc["A"]["recall"] == 0.5
    assert pc["B"]["precision"] == pytest.approx(2 / 3) and pc["B"]["recall"] == 1.0
    f_a, f_b = 2 / 3, 0.8
    # C has no support and is never predicted: precision/recall/F1 undefined -> None, not zero
    assert pc["C"]["precision"] is None and pc["C"]["f1"] is None and pc["C"]["support"] == 0
    assert m["macro_f1"] == pytest.approx((f_a + f_b) / 2)
    assert m["weighted_f1"] == pytest.approx((2 * f_a + 2 * f_b) / 4)
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert pc["C"]["roc_auc_ovr"] is None and "roc_auc[C]" in m["undefined_metrics"]
    assert pc["A"]["roc_auc_ovr"] == pytest.approx(1.0)
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.8, 0.3, 0.8, 0.7])))
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert m["confusion_matrix_normalized"][2] == [None, None, None]


def test_log_loss_respects_non_alphabetical_class_order():
    """Regression: sklearn.log_loss assumes sorted label columns; canonical order is not sorted."""
    order = ["C", "A", "B"]
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
    y = ["C", "A", "B"]
    expected = -np.mean(np.log([0.7, 0.6, 0.6]))
    assert multiclass_log_loss(y, proba, order) == pytest.approx(expected)


def test_confusion_matrix_follows_class_order():
    m = evaluate_predictions(["B", "A"], ["A", "A"], np.array([[0.4, 0.6], [0.3, 0.7]]), ["B", "A"], "u", "t", "e")
    assert m["confusion_matrix"] == [[0, 1], [0, 1]]


def test_input_validation():
    with pytest.raises(ValueError):
        _eval(["A"], ["A"], np.ones((1, 2)) / 2)
    with pytest.raises(ValueError):
        _eval(["Z"], ["A"], np.ones((1, 3)) / 3)


def test_core_breakdown_counts_stretch_predictions_as_errors():
    out = core_breakdown(["A", "A", "C"], ["A", "C", "C"], ["A", "B"])
    assert out["n_samples"] == 2 and out["accuracy"] == 0.5 and out["predicted_outside_core"] == 1


def test_expected_calibration_error():
    ece, bins = expected_calibration_error(["A", "A"], ["A", "B"], [1.0, 1.0], n_bins=10)
    assert ece == pytest.approx(0.5)
    assert bins["count"].sum() == 2


def test_cluster_bootstrap_intervals():
    y = ["A", "B", "C"] * 10
    p_good, p_bad = list(y), ["A"] * 30
    groups = [i // 3 for i in range(30)]
    out = cluster_bootstrap(y, {"good": p_good, "bad": p_bad}, CLASSES, groups, n_resamples=200, seed=1)
    g = out["models"]["good"]["macro_f1"]
    assert g["estimate"] == 1.0 and g["ci_low"] == 1.0
    d = out["macro_f1_difference_good_minus_bad"]
    assert d["ci_low"] > 0 and d["share_of_resamples_positive"] == 1.0


def test_measure_latency_with_mock():
    calls = []
    summary, samples = measure_latency(lambda b: calls.append(len(b)), ["a", "b", "c"], batch_size=1,
                                       warmup=3, repeats=20)
    assert len(calls) == 23 and set(calls) == {1}
    assert len(samples) == 20 and summary["p95_ms"] >= summary["median_ms"] >= 0
    assert math.isfinite(summary["throughput_predictions_per_s"])


def test_summary_row_schema():
    row = summary_row(model_name="m", test_macro_f1=0.5)
    assert list(row) == SUMMARY_FIELDS and row["test_accuracy"] is None
    with pytest.raises(ValueError):
        summary_row(not_a_field=1)
