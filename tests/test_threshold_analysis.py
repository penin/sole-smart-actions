"""Threshold analysis: decision boundaries, fallback mapping, metric arithmetic and V1 integrity."""

import numpy as np
import pandas as pd
import pytest

from sole_intent.rejection import (FALLBACK_ACTION, UNKNOWN_INTENT, application_action, application_response,
                                   apply_rejection)
from sole_intent.threshold_analysis import (assert_safe_output_dir, known_outcomes, resolve_thresholds,
                                            run_threshold_analysis, threshold_metrics, threshold_tag,
                                            unknown_outcomes, v1_manifest)
from sole_intent.utils import PROJECT_ROOT, load_json

LABELS = ["TRANSFER_MONEY", "BILL_PAY", "HELP_SUPPORT", "CHECK_BALANCE"]
FIN = ["TRANSFER_MONEY", "BILL_PAY", "WITHDRAW_MONEY"]


def row(top: int, p: float, n: int = len(LABELS)) -> list[float]:
    """Probability row with p on class index `top` and the rest spread evenly (p must exceed 1/n to be the max)."""
    rest = (1 - p) / (n - 1)
    assert p > rest, "fixture: p would not be the maximum"
    r = [rest] * n
    r[top] = p
    return r


NINE = [f"I{i}" for i in range(9)]  # same class count as the real model


# ------------------------------------------------------------------ boundaries and decisions
@pytest.mark.parametrize("t", [0.196, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60])
def test_threshold_boundary_conditions(t):
    proba = np.array([row(0, t, 9), row(0, np.nextafter(t, 0), 9), row(0, np.nextafter(t, 1), 9)])
    out = apply_rejection(proba, NINE, t)
    assert list(out) == ["I0", UNKNOWN_INTENT, "I0"]  # exactly equal -> accepted; just below -> rejected


def test_unknown_intent_assignment_and_top_intent_preserved():
    proba = np.array([row(1, 0.34), row(1, 0.36), row(2, 0.90)])
    top, decision, _ = unknown_outcomes(proba, LABELS, 0.35, FIN)
    assert list(top) == ["BILL_PAY", "BILL_PAY", "HELP_SUPPORT"]
    assert list(decision) == [UNKNOWN_INTENT, "BILL_PAY", "HELP_SUPPORT"]


def test_decisions_consistent_across_thresholds():
    """Raising the threshold can only turn acceptances into rejections; it never changes the routed intent."""
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(len(LABELS)), size=500)
    argmax = np.asarray(LABELS, dtype=object)[proba.argmax(1)]
    prev_rejected = np.zeros(500, dtype=bool)
    for t in [0.196, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        d = apply_rejection(proba, LABELS, t)
        rejected = d == UNKNOWN_INTENT
        assert np.all(rejected >= prev_rejected)
        assert np.all(d[~rejected] == argmax[~rejected])
        assert np.array_equal(rejected, proba.max(1) < t)
        prev_rejected = rejected


# ------------------------------------------------------------------ application mapping
def test_fallback_mapping_is_separate_from_help_support():
    assert application_action(UNKNOWN_INTENT) == FALLBACK_ACTION
    assert application_action("HELP_SUPPORT") == "ROUTE_HELP_SUPPORT"
    assert application_action("HELP_SUPPORT") != application_action(UNKNOWN_INTENT)
    fb = application_response(UNKNOWN_INTENT, "BILL_PAY")
    assert fb["action"] == FALLBACK_ACTION and fb["model_decision"] == UNKNOWN_INTENT
    assert fb["route_to"] is None and "HELP_SUPPORT" in fb["offers"] and fb["suggested_intent"] == "BILL_PAY"
    help_ = application_response("HELP_SUPPORT")
    assert help_["route_to"] == "HELP_SUPPORT" and help_["model_decision"] == "HELP_SUPPORT" and not help_["offers"]


# ------------------------------------------------------------------ metric arithmetic
@pytest.fixture
def fixture_sets():
    known_true = ["TRANSFER_MONEY", "TRANSFER_MONEY", "BILL_PAY", "HELP_SUPPORT", "CHECK_BALANCE"]
    known = np.array([row(0, 0.90),   # correct accept
                      row(1, 0.50),   # incorrect accept (TRANSFER -> BILL_PAY: financial misroute)
                      row(1, 0.28),   # false reject
                      row(2, 0.30),   # false reject
                      row(3, 0.80)])  # correct accept
    unknown = np.array([row(0, 0.28),  # rejected
                        row(2, 0.60),  # HELP_SUPPORT
                        row(0, 0.70),  # TRANSFER (financial)
                        row(3, 0.40)])  # CHECK_BALANCE (other)
    return known_true, known, unknown


def test_threshold_metrics_hand_computed(fixture_sets):
    known_true, known, unknown = fixture_sets
    m = threshold_metrics(known_true, known, unknown, LABELS, 0.35, FIN, unknown_groups=["a", "b", "c", "c"])
    k, u, a = m["model_classification"]["known"], m["model_classification"]["unknown"], m["application"]
    assert (k["correct_accept"], k["incorrect_accept"], k["false_reject"]) == (2, 1, 2)
    assert k["coverage"] == pytest.approx(3 / 5) and k["false_rejection_rate"] == pytest.approx(2 / 5)
    assert k["accepted_accuracy"] == pytest.approx(2 / 3) and k["end_to_end_accuracy"] == pytest.approx(2 / 5)
    assert k["misrouted_to_financial"] == 1
    assert k["false_rejection_by_intent"]["BILL_PAY"] == {"support": 1, "false_rejected": 1, "false_rejection_rate": 1.0}
    assert k["false_rejection_by_intent"]["TRANSFER_MONEY"]["false_rejected"] == 0
    assert u["rejected_as_unknown_intent"] == 1 and u["rejection_rate"] == pytest.approx(0.25)
    assert u["false_accept_rate"] == pytest.approx(0.75)
    assert u["classified_as_help_support"] == 1 and u["classified_as_financial"] == 1
    assert u["classified_as_financial_by_intent"] == {"TRANSFER_MONEY": 1, "BILL_PAY": 0, "WITHDRAW_MONEY": 0}
    assert u["classified_as_other_intent"] == 1
    assert u["rejection_rate_unique_phrases"] == pytest.approx((1 + 0 + 0) / 3)
    assert a["unknown_safe_fallback"] == 2 and a["unknown_safe_fallback_rate"] == pytest.approx(0.5)
    assert a["known_fallback_triggered"] == 2


def test_accepted_macro_f1_hand_computed(fixture_sets):
    known_true, known, unknown = fixture_sets
    k = threshold_metrics(known_true, known, unknown, LABELS, 0.35, FIN)["model_classification"]["known"]
    # accepted rows: TRANSFER->TRANSFER, TRANSFER->BILL_PAY, CHECK_BALANCE->CHECK_BALANCE
    # F1: TRANSFER 2*1*0.5/1.5=2/3, BILL_PAY 0 (predicted once, never true), CHECK_BALANCE 1; HELP undefined
    assert k["accepted_macro_f1"] == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert k["accepted_macro_f1_defined_classes"] == 3


def test_extreme_thresholds(fixture_sets):
    known_true, known, unknown = fixture_sets
    lo = threshold_metrics(known_true, known, unknown, LABELS, 0.0, FIN)
    hi = threshold_metrics(known_true, known, unknown, LABELS, 1.01, FIN)
    assert lo["model_classification"]["known"]["coverage"] == 1.0
    assert lo["model_classification"]["unknown"]["rejection_rate"] == 0.0
    assert hi["model_classification"]["known"]["coverage"] == 0.0
    assert hi["model_classification"]["known"]["accepted_accuracy"] is None  # undefined, not zero
    assert hi["application"]["unknown_safe_fallback_rate"] == 1.0


def test_outcome_labels_cover_every_row(fixture_sets):
    known_true, known, unknown = fixture_sets
    _, _, ko = known_outcomes(known_true, known, LABELS, 0.35)
    _, _, uo = unknown_outcomes(unknown, LABELS, 0.35, FIN)
    assert list(ko) == ["correct_accept", "incorrect_accept", "false_reject", "false_reject", "correct_accept"]
    assert list(uo) == ["correct_reject", "accepted_as_help_support", "accepted_as_financial", "accepted_as_other_intent"]


def test_helpers():
    assert threshold_tag(0.35) == "035" and threshold_tag(0.6) == "060"
    assert resolve_thresholds(["baseline", 0.35, "0.25", 0.35], 0.1957) == [0.1957, 0.25, 0.35]


# ------------------------------------------------------------------ integrity on the real project
def test_refuses_to_write_into_v1_folders():
    for d in ["results/history/bow_v4/metrics", "results/history/bow_v4/predictions", "results/history/bow_v4/reports",
              "results/history/bow_v4/figures", "artifacts/history/bow_v4"]:
        with pytest.raises(ValueError):
            assert_safe_output_dir(PROJECT_ROOT / d)
        with pytest.raises(ValueError):
            assert_safe_output_dir(PROJECT_ROOT / d / "sub")
    assert_safe_output_dir(PROJECT_ROOT / "results" / "history" / "bow_v4" / "threshold_analysis")


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    before = v1_manifest()
    ctx = run_threshold_analysis(output_dir=tmp_path_factory.mktemp("ta"), make_plots=False)
    return before, ctx


def test_analysis_does_not_modify_v1_artifacts(analysis):
    before, ctx = analysis
    assert v1_manifest() == before
    assert ctx["v1_unchanged"] is True
    model = PROJECT_ROOT / "artifacts" / "history" / "bow_v4" / "bow_logreg_v1_tuned_v1.0.0.joblib"
    assert ctx["payload"]["model_sha256"] == before["artifacts/history/bow_v4/bow_logreg_v1_tuned_v1.0.0.joblib"]
    assert model.exists()


def test_analysis_reproduces_v1_and_uses_saved_probabilities(analysis):
    _, ctx = analysis
    v1 = load_json(PROJECT_ROOT / "results/history/bow_v4/metrics/rejection_metrics.json")["primary_frozen_threshold_test_plus_unknown"]
    row0 = ctx["table"][ctx["table"]["is_baseline_v1"]].iloc[0]
    assert row0["threshold"] == pytest.approx(v1["threshold"])
    assert row0["known_coverage"] == pytest.approx(v1["known_intent_coverage"])
    assert row0["unknown_rejection_rate"] == pytest.approx(v1["unknown_rejection_rate"])
    assert ctx["payload"]["probability_source"] == "saved_v1_predictions"
    assert ctx["payload"]["max_abs_diff_saved_vs_model_probabilities"] < 1e-9


def test_analysis_table_is_consistent(analysis):
    _, ctx = analysis
    t = ctx["table"]
    assert t["threshold"].is_monotonic_increasing
    assert t["known_coverage"].is_monotonic_decreasing and t["unknown_rejection_rate"].is_monotonic_increasing
    np.testing.assert_allclose(t["known_coverage"] + t["known_false_rejection_rate"], 1)
    np.testing.assert_allclose(t["unknown_rejection_rate"] + t["unknown_false_accept_rate"], 1)
    np.testing.assert_allclose(t["safe_fallback_rate"],
                               (t["unknown_rejected"] + t["unknown_to_help_support"]) / t["unknown_n"])
    parts = t["unknown_rejected"] + t["unknown_to_help_support"] + t["unknown_to_financial"] + t["unknown_to_other_intents"]
    assert (parts == t["unknown_n"]).all()
    per_intent = t[[c for c in t.columns if c.startswith("false_rejected_")]].sum(axis=1)
    assert (per_intent == t["known_false_reject"]).all()
    assert set(t["threshold"].round(3)) >= {0.196, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6}


def test_candidate_exports(analysis):
    _, ctx = analysis
    out = ctx["out_dir"]
    preds = pd.read_csv(out / "threshold_035_predictions.csv")
    assert len(preds) == 270 + 300
    assert ((preds["model_decision"] == UNKNOWN_INTENT) == (preds["max_confidence"] < 0.35)).all()
    assert ((preds["model_decision"] == UNKNOWN_INTENT) == (preds["application_action"] == FALLBACK_ACTION)).all()
    errors = pd.read_csv(out / "threshold_035_errors.csv")
    assert not errors["outcome"].isin(["correct_accept", "correct_reject"]).any()
    assert (out / "threshold_035_report.md").exists()
