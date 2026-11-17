"""Inference with a saved model bundle, including confidence-based rejection.

Classification only: this module never executes payments or touches live wallet systems.

    python -m sole_intent.predict "send $50 to Jone" "who won the rugby"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .rejection import UNKNOWN_INTENT, application_response, apply_rejection
from .train import load_bundle, reorder_proba
from .utils import MODELS_DIR, load_config, resolve_path


def default_model_path() -> Path:
    candidates = sorted(MODELS_DIR.glob("*_tuned_v*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No tuned model in {MODELS_DIR}; run `python scripts/run_all.py` first")
    return candidates[-1]


def predict_texts(texts: list[str], bundle: dict, threshold: float | None = None) -> list[dict]:
    class_order = bundle["class_order"]
    thr = bundle["rejection_threshold"] if threshold is None else threshold
    proba = reorder_proba(bundle["pipeline"], texts, class_order)
    raw = np.asarray(class_order, dtype=object)[proba.argmax(axis=1)]
    final = apply_rejection(proba, class_order, thr) if thr is not None else raw
    return [{
        "text": t,
        "predicted_intent": str(raw[i]),
        "final_intent": str(final[i]),
        "rejected": bool(final[i] == UNKNOWN_INTENT),
        "max_confidence": float(proba[i].max()),
        "threshold": thr,
        "probabilities": {c: float(p) for c, p in zip(class_order, proba[i])},
        "application": application_response(str(final[i]), str(raw[i])),
    } for i, t in enumerate(texts)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Sole Smart Actions commands.")
    parser.add_argument("texts", nargs="+", help="One or more commands to classify")
    parser.add_argument("--model", type=Path, default=None, help="Path to a .joblib bundle")
    parser.add_argument("--threshold", type=float, default=None, help="Override the rejection threshold")
    parser.add_argument("--config", default=None, help="Inference YAML (e.g. config/history/bow_v4_inference.yaml)")
    parser.add_argument("--json", action="store_true", help="Print full JSON including probabilities")
    args = parser.parse_args()
    inference = load_config(args.config).get("inference", {}) if args.config else {}
    model_path = args.model or (resolve_path(inference["model_path"]) if inference.get("model_path")
                                else default_model_path())
    threshold = args.threshold if args.threshold is not None else inference.get("threshold")
    bundle = load_bundle(model_path)
    results = predict_texts(args.texts, bundle, threshold)
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        print(f"{r['final_intent']:<18} conf={r['max_confidence']:.3f} thr={r['threshold']:.3f}  "
              f"app={r['application']['action']:<26} (top: {r['predicted_intent']})  {r['text']!r}")


if __name__ == "__main__":
    main()
