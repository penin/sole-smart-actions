import numpy as np
import pytest

from sole_intent.rejection import (UNKNOWN_INTENT, apply_rejection, balanced_threshold, max_confidence,
                                   rejection_outcomes, threshold_for_coverage, threshold_sweep)

CLASSES = ["A", "B", "C"]


def test_apply_rejection_rule(toy_proba):
    out = apply_rejection(toy_proba, CLASSES, threshold=0.5)
    assert list(out) == ["A", UNKNOWN_INTENT, "C", UNKNOWN_INTENT]


def test_threshold_boundary_is_inclusive(toy_proba):
    # max prob exactly equal to the threshold is accepted (rule rejects only when strictly below)
    out = apply_rejection(toy_proba, CLASSES, threshold=0.7)
    assert out[0] == "A" and out[2] == "C" and out[1] == UNKNOWN_INTENT


def test_threshold_extremes(toy_proba):
    assert UNKNOWN_INTENT not in apply_rejection(toy_proba, CLASSES, 0.0)
    assert set(apply_rejection(toy_proba, CLASSES, 1.01)) == {UNKNOWN_INTENT}


def test_apply_rejection_shape_check(toy_proba):
    with pytest.raises(ValueError):
        apply_rejection(toy_proba, ["A", "B"], 0.5)


def test_threshold_for_coverage():
    conf = np.linspace(0.1, 1.0, 10)
    thr = threshold_for_coverage(conf, 0.9)
    assert thr == pytest.approx(0.2)
    assert (conf >= thr).mean() >= 0.9
    assert threshold_for_coverage(conf, 1.0) == pytest.approx(0.1)
    rng = np.random.default_rng(0)
    for target in (0.8, 0.95, 0.99):
        c = rng.random(273)
        assert (c >= threshold_for_coverage(c, target)).mean() >= target
    with pytest.raises(ValueError):
        threshold_for_coverage([], 0.9)
    with pytest.raises(ValueError):
        threshold_for_coverage(conf, 0)


def test_rejection_outcomes_counts(toy_proba):
    known_true = ["A", "A", "C", "B"]          # row 2 predicted A but low conf; row 4 low conf
    unknown = np.array([[0.9, 0.05, 0.05], [0.34, 0.33, 0.33]])
    o = rejection_outcomes(known_true, toy_proba, unknown, CLASSES, threshold=0.5)
    c = o["counts"]
    assert c == {"correct_supported_classification": 2, "incorrect_supported_classification": 0,
                 "false_rejection_of_supported": 2, "correct_rejection_of_unsupported": 1,
                 "false_acceptance_of_unsupported": 1}
    assert o["known_intent_coverage"] == 0.5
    assert o["unknown_rejection_rate"] == 0.5 and o["unknown_false_accept_rate"] == 0.5
    assert o["unknown_false_accepts_by_predicted_intent"] == {"A": 1}


def test_rejection_outcomes_without_unknowns(toy_proba):
    o = rejection_outcomes(["A", "A", "C", "B"], toy_proba, np.empty((0, 3)), CLASSES, 0.5)
    assert o["unknown_rejection_rate"] is None  # undefined, not zero


def test_sweep_is_monotonic(toy_proba):
    unknown = np.array([[0.5, 0.3, 0.2], [0.34, 0.33, 0.33]])
    sweep = threshold_sweep(["A", "A", "C", "B"], toy_proba, unknown, CLASSES, np.linspace(0, 1, 21))
    assert sweep["known_intent_coverage"].is_monotonic_decreasing
    assert sweep["unknown_rejection_rate"].is_monotonic_increasing


def test_balanced_threshold_separates():
    known = np.array([[0.9, 0.05, 0.05]] * 10)
    unknown = np.array([[0.4, 0.3, 0.3]] * 10)
    t = balanced_threshold(known, unknown, np.round(np.linspace(0, 1, 101), 2))
    assert 0.4 < t <= 0.9
    assert (max_confidence(known) >= t).all() and (max_confidence(unknown) < t).all()


def test_real_model_rejects_with_frozen_threshold(fitted_pipeline, splits, labels):
    from sole_intent.train import reorder_proba
    val_conf = max_confidence(reorder_proba(fitted_pipeline, splits.validation["text"], labels))
    thr = threshold_for_coverage(val_conf, 0.95)
    assert (val_conf >= thr).mean() >= 0.95
    final = apply_rejection(reorder_proba(fitted_pipeline, splits.unknown["text"], labels), labels, thr)
    assert set(final) <= set(labels) | {UNKNOWN_INTENT}
