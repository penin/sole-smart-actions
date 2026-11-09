"""Confidence-threshold rejection of unsupported (UNKNOWN_INTENT) requests.

Rule: if max class probability < threshold -> UNKNOWN_INTENT, else the argmax intent.
This is a baseline out-of-distribution heuristic, not a guarantee of detection.
Model-agnostic: works on any (n_samples, n_classes) probability matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNKNOWN_INTENT = "UNKNOWN_INTENT"

# Application layer: what the app does with a model decision. Kept separate from the model output so an
# internal UNKNOWN_INTENT is never confused with a genuine HELP_SUPPORT classification.
FALLBACK_ACTION = "FALLBACK_HELP_OR_CLARIFY"
FALLBACK_OFFERS = ("HELP_SUPPORT", "CLARIFY")


def application_action(model_decision: str) -> str:
    """Map a model decision to an app action: ROUTE_<INTENT>, or the fallback for UNKNOWN_INTENT."""
    return FALLBACK_ACTION if model_decision == UNKNOWN_INTENT else f"ROUTE_{model_decision}"


def application_response(model_decision: str, top_intent: str | None = None) -> dict:
    """App-facing response. A fallback offers help/clarification and may suggest the model's top intent;
    it never executes that intent."""
    if model_decision == UNKNOWN_INTENT:
        return {"action": FALLBACK_ACTION, "model_decision": UNKNOWN_INTENT, "route_to": None,
                "offers": list(FALLBACK_OFFERS), "suggested_intent": top_intent}
    return {"action": application_action(model_decision), "model_decision": model_decision,
            "route_to": model_decision, "offers": [], "suggested_intent": None}


def max_confidence(proba) -> np.ndarray:
    return np.asarray(proba, dtype=float).max(axis=1)


def apply_rejection(proba, class_order: list[str], threshold: float,
                    unknown_label: str = UNKNOWN_INTENT) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != len(class_order):
        raise ValueError(f"proba shape {proba.shape} incompatible with {len(class_order)} classes")
    predicted = np.asarray(class_order, dtype=object)[proba.argmax(axis=1)]
    return np.where(proba.max(axis=1) < threshold, unknown_label, predicted).astype(object)


def threshold_for_coverage(confidences, target_coverage: float) -> float:
    """Largest observed-confidence threshold that still accepts >= target_coverage of the
    (known-intent) examples. Uses known-intent data only; unknown labels are not consulted."""
    conf = np.sort(np.asarray(confidences, dtype=float))
    if conf.size == 0:
        raise ValueError("No confidences supplied")
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    k = int(np.floor((1 - target_coverage) * conf.size + 1e-9))
    return float(conf[min(k, conf.size - 1)])


def rejection_outcomes(known_true, known_proba, unknown_proba, class_order: list[str],
                       threshold: float) -> dict:
    """Five-way outcome breakdown for a combined known + unknown evaluation set."""
    known_true = np.asarray(known_true, dtype=object)
    known_final = apply_rejection(known_proba, class_order, threshold)
    unknown_final = apply_rejection(unknown_proba, class_order, threshold)
    n_known, n_unknown = len(known_true), len(unknown_final)

    known_rejected = known_final == UNKNOWN_INTENT
    correct_accept = (~known_rejected) & (known_final == known_true)
    wrong_accept = (~known_rejected) & (known_final != known_true)
    unknown_rejected = unknown_final == UNKNOWN_INTENT

    def rate(num, den):
        return float(num / den) if den else None

    accepted = int((~known_rejected).sum())
    out = {
        "threshold": float(threshold),
        "n_known": n_known,
        "n_unknown": n_unknown,
        "counts": {
            "correct_supported_classification": int(correct_accept.sum()),
            "incorrect_supported_classification": int(wrong_accept.sum()),
            "false_rejection_of_supported": int(known_rejected.sum()),
            "correct_rejection_of_unsupported": int(unknown_rejected.sum()),
            "false_acceptance_of_unsupported": int((~unknown_rejected).sum()),
        },
        "known_intent_coverage": rate(accepted, n_known),
        "known_false_rejection_rate": rate(int(known_rejected.sum()), n_known),
        "accuracy_on_accepted_known": rate(int(correct_accept.sum()), accepted),
        "end_to_end_known_accuracy": rate(int(correct_accept.sum()), n_known),
        "unknown_rejection_rate": rate(int(unknown_rejected.sum()), n_unknown),
        "unknown_false_accept_rate": rate(int((~unknown_rejected).sum()), n_unknown),
    }
    if n_unknown:
        accepted_as = pd.Series(unknown_final[~unknown_rejected]).value_counts().to_dict()
        out["unknown_false_accepts_by_predicted_intent"] = {str(k): int(v) for k, v in accepted_as.items()}
    return out


def threshold_sweep(known_true, known_proba, unknown_proba, class_order: list[str],
                    thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        o = rejection_outcomes(known_true, known_proba, unknown_proba, class_order, float(t))
        rows.append({"threshold": float(t),
                     "known_intent_coverage": o["known_intent_coverage"],
                     "known_false_rejection_rate": o["known_false_rejection_rate"],
                     "accuracy_on_accepted_known": o["accuracy_on_accepted_known"],
                     "unknown_rejection_rate": o["unknown_rejection_rate"],
                     "unknown_false_accept_rate": o["unknown_false_accept_rate"]})
    return pd.DataFrame(rows)


def balanced_threshold(known_proba, unknown_proba, thresholds) -> float:
    """Unknown-aware threshold maximizing 0.5*(known coverage) + 0.5*(unknown rejection).
    Only to be used with a *calibration* unknown set, never with the set it is reported on."""
    k = max_confidence(known_proba)
    u = max_confidence(unknown_proba)
    scores = [(0.5 * (k >= t).mean() + 0.5 * (u < t).mean(), -t) for t in thresholds]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return float(thresholds[best])
