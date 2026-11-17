"""Confidence thresholds: validation calibration, the standard sweep, and secondary routing analyses.

Confidence = maximum class probability. A command is *accepted* (routed to its predicted intent) when
confidence >= threshold; otherwise it becomes UNKNOWN_INTENT and the app offers help / clarification.
UNKNOWN_INTENT and HELP_SUPPORT are reported separately even though both count as safe fallbacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from sole_intent.rejection import threshold_for_coverage


def calibrate_threshold(validation_proba: np.ndarray, target_coverage: float) -> dict:
    """The primary default threshold: uses validation known-intent confidences only (no unknown labels)."""
    conf = np.asarray(validation_proba).max(axis=1)
    t = threshold_for_coverage(conf, target_coverage)
    return {"algorithm": "largest observed validation max-probability with coverage >= target "
                         "(sole_intent.rejection.threshold_for_coverage)",
            "confidence": "max_probability", "calibration_split": "validation", "n_calibration": int(conf.size),
            "target_known_coverage": float(target_coverage), "threshold": float(t),
            "actual_validation_coverage": float((conf >= t).mean()),
            "uses_unknown_labels": False, "uses_test_labels": False}


def _resolve(spec, default: float) -> tuple[str, float]:
    if spec in (None, "none"):
        return "none", 0.0
    if spec == "default":
        return "default", float(default)
    return f"{float(spec):.2f}", float(spec)


def threshold_row(label: str, t: float, y_true, proba, unknown_proba, class_order, financial, help_intent,
                  validation_proba=None) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    classes = np.asarray(class_order, dtype=object)
    conf, pred = proba.max(1), classes[proba.argmax(1)]
    acc = conf >= t
    correct_acc = acc & (pred == y_true)
    uconf, upred = unknown_proba.max(1), classes[unknown_proba.argmax(1)]
    uacc = uconf >= t
    u_help = uacc & (upred == help_intent)
    u_fin = uacc & np.isin(upred, list(financial))
    n, nu = len(y_true), len(upred)
    row = {
        "threshold_label": label, "threshold": t,
        "validation_known_coverage": float((validation_proba.max(1) >= t).mean()) if validation_proba is not None else None,
        # 1-6: legitimate test commands (n = 270)
        "known_coverage": float(acc.mean()),
        "accepted_known": int(acc.sum()),
        "accepted_known_correct": int(correct_acc.sum()),
        "accuracy_among_accepted": float(correct_acc.sum() / acc.sum()) if acc.sum() else None,
        "overall_correct_routing_rate": float(correct_acc.sum() / n),
        "rejected_known": int((~acc).sum()),
        "accepted_known_wrong": int((acc & (pred != y_true)).sum()),
        "accepted_known_wrong_financial": int((acc & (pred != y_true) & np.isin(pred, list(financial))).sum()),
        # 7-10: unknown-evaluation commands (n = 300)
        "unknown_rejection_rate": float((~uacc).mean()),
        "unknown_rejected": int((~uacc).sum()),
        "unknown_as_help_support": int(u_help.sum()),
        "unknown_as_help_support_rate": float(u_help.mean()),
        "unknown_safe_fallback_rate": float(((~uacc) | u_help).mean()),
        "unknown_to_financial": int(u_fin.sum()),
        "unknown_to_financial_rate": float(u_fin.mean()),
        "unknown_to_other_intent": int((uacc & ~(upred == help_intent) & ~np.isin(upred, list(financial))).sum()),
        "n_known": n, "n_unknown": nu,
    }
    return row


def threshold_sweep(y_true, proba, unknown_proba, class_order, specs, default: float, financial,
                    help_intent: str, validation_proba=None) -> pd.DataFrame:
    rows = []
    for spec in specs:
        label, t = _resolve(spec, default)
        rows.append(threshold_row(label, t, y_true, proba, unknown_proba, class_order, financial, help_intent,
                                  validation_proba))
    return pd.DataFrame(rows)


def dense_sweep(y_true, proba, unknown_proba, class_order, financial, help_intent, validation_proba,
                step: float = 0.01) -> pd.DataFrame:
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    return pd.DataFrame([threshold_row(f"{t:.2f}", float(t), y_true, proba, unknown_proba, class_order, financial,
                                       help_intent, validation_proba) for t in grid])


# ------------------------------------------------------------------ secondary analyses
def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """The teammates' temperature scaling: divide log-probabilities by T, renormalise."""
    logits = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def _nll(proba, y_idx) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-7, 1.0))))


def ece(proba, y_idx, bins: int = 10) -> float:
    conf, correct = proba.max(1), proba.argmax(1) == y_idx
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(err)


def temperature_calibration(val_proba, val_idx, test_proba, test_idx) -> dict:
    """Fit T on validation log loss (as both neural teammates did) and report before/after calibration."""
    res = minimize_scalar(lambda lt: _nll(apply_temperature(val_proba, np.exp(lt)), val_idx), bounds=(-4, 4),
                          method="bounded")
    T = float(np.exp(res.x))
    out = {"temperature": T, "fitted_on": "validation (known intents only)"}
    for split, p, y in (("validation", val_proba, val_idx), ("test", test_proba, test_idx)):
        cal = apply_temperature(p, T)
        out[split] = {"log_loss_raw": _nll(p, y), "log_loss_calibrated": _nll(cal, y),
                      "ece_raw": ece(p, y), "ece_calibrated": ece(cal, y),
                      "mean_confidence_raw": float(p.max(1).mean()), "mean_confidence_calibrated": float(cal.max(1).mean())}
    out["note"] = ("Temperature scaling never changes the predicted intent. The validation figures are measured on "
                   "the same commands used to fit T (optimistic); test figures are held out.")
    return out


def three_tier_routing(y_true, proba, unknown_proba, class_order, open_t: float, menu_t: float,
                       menu_size: int, financial) -> dict:
    """Ricardo's three-tier gate, with both thresholds calibrated on validation (no unknown labels):
    >= open_t open the workflow; [menu_t, open_t) show the top-k intents + 'something else'; < menu_t not understood."""
    classes = np.asarray(class_order, dtype=object)
    y_true = np.asarray(y_true, dtype=object)

    def tiers(p):
        conf = p.max(1)
        return np.where(conf >= open_t, "open", np.where(conf >= menu_t, "menu", "not_understood"))

    t, tu = tiers(proba), tiers(unknown_proba)
    pred, upred = classes[proba.argmax(1)], classes[unknown_proba.argmax(1)]
    topk = np.argsort(-proba, axis=1)[:, :menu_size]
    in_menu = np.array([class_order.index(y) in row for y, row in zip(y_true, topk)])
    n, nu = len(y_true), len(upred)
    return {
        "open_threshold": float(open_t), "menu_threshold": float(menu_t), "menu_size": menu_size,
        "known": {"open_correct": float(((t == "open") & (pred == y_true)).sum() / n),
                  "open_wrong": float(((t == "open") & (pred != y_true)).sum() / n),
                  "open_wrong_financial": float(((t == "open") & (pred != y_true) & np.isin(pred, list(financial))).sum() / n),
                  "menu_contains_intent": float(((t == "menu") & in_menu).sum() / n),
                  "menu_missing_intent": float(((t == "menu") & ~in_menu).sum() / n),
                  "not_understood": float((t == "not_understood").sum() / n)},
        "unknown": {"opened": float((tu == "open").sum() / nu),
                    "opened_financial": float(((tu == "open") & np.isin(upred, list(financial))).sum() / nu),
                    "menu": float((tu == "menu").sum() / nu),
                    "not_understood": float((tu == "not_understood").sum() / nu)},
    }
