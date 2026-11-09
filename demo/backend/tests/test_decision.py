"""Threshold/fallback decision: boundaries, rule preservation and the shared frontend fixture."""

import json

import numpy as np
import pytest

from decision import apply_threshold
from generate_threshold_cases import FIXTURE, build_cases
from sole_intent.rejection import FALLBACK_ACTION, UNKNOWN_INTENT, apply_rejection

ORDER = ["A", "B", "HELP_SUPPORT"]


def probs(a, b, h):
    return {"A": a, "B": b, "HELP_SUPPORT": h}


def test_accept_and_fallback():
    d = apply_threshold(probs(0.7, 0.2, 0.1), 0.3, ORDER)
    assert (d["raw_prediction"], d["final_intent"], d["status"], d["fallback_applied"]) == ("A", "A", "accepted", False)
    d = apply_threshold(probs(0.7, 0.2, 0.1), 0.8, ORDER)
    assert d["raw_prediction"] == "A" and d["confidence"] == 0.7
    assert d["final_intent"] == UNKNOWN_INTENT and d["status"] == "fallback_applied" and d["below_threshold"]
    assert d["application"]["action"] == FALLBACK_ACTION and "HELP_SUPPORT" in d["application"]["offers"]
    assert d["application"]["suggested_intent"] == "A"


def test_confidence_equal_to_threshold_is_accepted():
    assert apply_threshold(probs(0.35, 0.33, 0.32), 0.35, ORDER)["final_intent"] == "A"
    assert apply_threshold(probs(0.35, 0.33, 0.32), float(np.nextafter(0.35, 1)), ORDER)["final_intent"] == UNKNOWN_INTENT


@pytest.mark.parametrize("t", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_full_slider_range(t):
    d = apply_threshold(probs(0.5, 0.3, 0.2), t, ORDER)
    assert d["final_intent"] == ("A" if t <= 0.5 else UNKNOWN_INTENT)


def test_help_support_prediction_is_routed_not_fallback():
    d = apply_threshold(probs(0.1, 0.2, 0.7), 0.5, ORDER)
    assert d["final_intent"] == "HELP_SUPPORT" and not d["fallback_applied"]
    assert d["application"]["action"] == "ROUTE_HELP_SUPPORT"


def test_low_confidence_is_not_silently_relabelled_help_support():
    d = apply_threshold(probs(0.4, 0.35, 0.25), 0.5, ORDER)
    assert d["final_intent"] == UNKNOWN_INTENT != "HELP_SUPPORT"


def test_probabilities_not_modified():
    p = probs(0.6, 0.3, 0.1)
    before = dict(p)
    for t in np.linspace(0, 1, 11):
        apply_threshold(p, t, ORDER)
    assert p == before


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
def test_invalid_threshold(bad):
    with pytest.raises(ValueError):
        apply_threshold(probs(0.6, 0.3, 0.1), bad, ORDER)


def test_class_mismatch_rejected():
    with pytest.raises(ValueError):
        apply_threshold({"A": 1.0}, 0.5, ORDER)


def test_matches_existing_project_rule_on_random_inputs():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(len(ORDER)), size=300)
    for t in [0.0, 0.196, 0.35, 0.5, 1.0]:
        expected = apply_rejection(P, ORDER, t)
        got = [apply_threshold(dict(zip(ORDER, row)), t, ORDER)["final_intent"] for row in P]
        assert list(expected) == got


def test_shared_frontend_fixture_is_current():
    """The fixture the frontend tests run against must equal what the backend produces now."""
    on_disk = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(build_cases()))
    names = [c["name"] for c in on_disk["cases"]]
    assert any("(equal)" in n for n in names) and any("+ 1ulp" in n for n in names)
