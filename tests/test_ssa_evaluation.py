"""Shared evaluation framework: metric definitions, thresholds, unknown routing and safe fallback."""

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from sole_intent.evaluate import evaluate_predictions
from sole_smart_actions.evaluation.thresholds import (apply_temperature, calibrate_threshold, threshold_row,
                                                      threshold_sweep, three_tier_routing)

C = ["A", "B", "HELP_SUPPORT", "TRANSFER_MONEY"]
FIN = ["TRANSFER_MONEY"]


def onehot(idx, conf):
    p = np.full((len(idx), len(C)), 0.0)
    for r, (i, c) in enumerate(zip(idx, conf)):
        p[r] = (1 - c) / (len(C) - 1)
        p[r, i] = c
    return p


def test_raw_accuracy_counts_every_command_and_ignores_thresholds():
    y = np.array(["A", "A", "B", "B", "HELP_SUPPORT"])
    p = onehot([0, 1, 1, 1, 2], [0.9, 0.4, 0.3, 0.99, 0.6])
    m = evaluate_predictions(y, np.array(C, dtype=object)[p.argmax(1)], p, C, "test", "x", "x")
    assert m["accuracy"] == 4 / 5
    sweep = threshold_sweep(y, p, p[:1], C, ["none", "default", 0.5], 0.5, FIN, "HELP_SUPPORT")
    none = sweep.iloc[0]
    assert none["overall_correct_routing_rate"] == m["accuracy"] and none["known_coverage"] == 1.0


def test_macro_f1_and_roc_auc_match_sklearn():
    rng = np.random.default_rng(0)
    y = np.array(C * 10, dtype=object)
    p = rng.dirichlet(np.ones(len(C)), size=len(y))
    pred = np.array(C, dtype=object)[p.argmax(1)]
    m = evaluate_predictions(y, pred, p, C, "test", "x", "x")
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, labels=C, average="macro", zero_division=0))
    # Canonical (non-alphabetical) class order must be respected
    sk_auc = np.mean([roc_auc_score(y == c, p[:, i]) for i, c in enumerate(C)])
    assert m["macro_roc_auc_ovr"] == pytest.approx(sk_auc)


def test_threshold_row_definitions():
    y = np.array(["A", "B", "A", "B"])
    p = onehot([0, 0, 0, 1], [0.9, 0.8, 0.3, 0.7])          # 2nd prediction is wrong, 3rd is low confidence
    u = onehot([2, 3, 0, 1, 3], [0.9, 0.9, 0.2, 0.9, 0.3])  # help, money, (rejected), other, (rejected)
    r = threshold_row("0.50", 0.5, y, p, u, C, FIN, "HELP_SUPPORT")
    assert r["known_coverage"] == 3 / 4 and r["accepted_known"] == 3 and r["accepted_known_correct"] == 2
    assert r["accuracy_among_accepted"] == 2 / 3 and r["overall_correct_routing_rate"] == 2 / 4
    assert r["rejected_known"] == 1
    assert r["unknown_rejected"] == 2 and r["unknown_rejection_rate"] == 2 / 5
    assert r["unknown_as_help_support"] == 1 and r["unknown_to_financial"] == 1 and r["unknown_to_other_intent"] == 1
    assert r["unknown_safe_fallback_rate"] == 3 / 5            # rejected OR accepted as HELP_SUPPORT
    assert r["unknown_as_help_support"] + r["unknown_rejected"] + r["unknown_to_financial"] + r["unknown_to_other_intent"] == 5


def test_equal_confidence_is_accepted():
    p = onehot([0], [0.5])
    r = threshold_row("x", 0.5, np.array(["A"]), p, p, C, FIN, "HELP_SUPPORT")
    assert r["accepted_known"] == 1


def test_calibration_uses_validation_only_and_meets_target():
    rng = np.random.default_rng(1)
    val = rng.dirichlet(np.ones(len(C)) * 0.3, size=270)
    cal = calibrate_threshold(val, 0.95)
    assert cal["actual_validation_coverage"] >= 0.95
    assert cal["uses_unknown_labels"] is False and cal["uses_test_labels"] is False
    assert cal["threshold"] in set(val.max(1))  # an observed confidence, not an interpolated value
    # the next larger observed confidence would drop below the target
    higher = np.sort(val.max(1))[np.searchsorted(np.sort(val.max(1)), cal["threshold"], side="right")]
    assert (val.max(1) >= higher).mean() < 0.95


def test_temperature_scaling_never_changes_the_prediction():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(len(C)), size=50)
    for t in (0.5, 1.0, 2.3):
        assert (apply_temperature(p, t).argmax(1) == p.argmax(1)).all()
        assert np.allclose(apply_temperature(p, t).sum(1), 1)


def test_three_tier_menu_contains_true_intent():
    y = np.array(["A", "B"])
    p = np.array([[0.4, 0.35, 0.15, 0.1], [0.97, 0.01, 0.01, 0.01]])
    out = three_tier_routing(y, p, p[:1], C, open_t=0.9, menu_t=0.3, menu_size=3, financial=FIN)
    assert out["known"]["menu_contains_intent"] == 0.5 and out["known"]["open_wrong"] == 0.5
