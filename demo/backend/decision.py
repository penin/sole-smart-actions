"""Threshold + fallback decision, separated from model inference.

Wraps the project's existing rules in ``sole_intent.rejection`` without changing them:

* model decision: ``UNKNOWN_INTENT`` if max probability < threshold, else the top intent
  (a probability exactly equal to the threshold is accepted; ties go to the first class in class order);
* application action: ``UNKNOWN_INTENT`` -> ``FALLBACK_HELP_OR_CLARIFY`` (offer HELP_SUPPORT or ask to
  rephrase); any intent -> ``ROUTE_<INTENT>``. A genuine HELP_SUPPORT prediction is routed, not a fallback.

The frontend mirrors this function for instant slider feedback; ``tests/test_threshold_cases.py`` and the
frontend's ``threshold.test.ts`` check both against the same generated fixture.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

import paths  # noqa: F401  (makes sole_intent importable)
from sole_intent.rejection import UNKNOWN_INTENT, application_response, apply_rejection


def validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if math.isnan(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return threshold


def apply_threshold(probabilities: Mapping[str, float], threshold: float, class_order: Sequence[str]) -> dict:
    """Final decision for one prediction. Probabilities are read, never modified."""
    threshold = validate_threshold(threshold)
    missing = [c for c in class_order if c not in probabilities]
    extra = [c for c in probabilities if c not in class_order]
    if missing or extra:
        raise ValueError(f"probabilities do not match model classes (missing={missing}, unexpected={extra})")
    proba = np.array([[float(probabilities[c]) for c in class_order]])
    final = str(apply_rejection(proba, list(class_order), threshold)[0])
    raw = str(class_order[int(proba[0].argmax())])
    accepted = final != UNKNOWN_INTENT
    return {
        "raw_prediction": raw,
        "confidence": float(proba[0].max()),
        "threshold": threshold,
        "final_intent": final,
        "status": "accepted" if accepted else "fallback_applied",
        "below_threshold": not accepted,
        "fallback_applied": not accepted,
        "application": application_response(final, raw),
    }
