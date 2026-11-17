"""Generate the shared threshold-decision and workflow fixtures used by BOTH test suites.

The backend's ``decision.apply_threshold`` (wrapping the project's ``apply_rejection``) produces the expected
results; the frontend's ``applyThreshold`` must reproduce them exactly (``src/__tests__/threshold.test.ts``).
Probabilities come from the real (primary) BoW model on every sample command, plus exact-boundary and tie cases.
``workflow_cases.json`` holds ``workflow.build_workflow`` outputs that ``src/lib/workflow.ts`` must reproduce.

    .venv\\Scripts\\python demo\\backend\\generate_threshold_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from decision import apply_threshold  # noqa: E402
from workflow import build_workflow  # noqa: E402
from model_service import BowLogRegModel  # noqa: E402

FIXTURE = paths.DEMO_DIR / "frontend" / "src" / "__tests__" / "fixtures" / "threshold_cases.json"
WORKFLOW_FIXTURE = FIXTURE.with_name("workflow_cases.json")
EXTRA_WORKFLOW_TEXTS = ["I owe Mary $50", "Send 20 to my mum", "Pay $75 to EFL", "Pay my bill", "Withdraw 100 dollars",
                        "Give Ana 20 dollars", "Transfer money", "Send money to The shop"]
GRID = [0.0, 0.01, 0.1, 0.25, 0.3, 0.35, 0.5, 0.6, 0.8, 0.99, 1.0]


def build_cases() -> dict:
    model = BowLogRegModel(paths.ARTIFACTS_DIR / "bow" / "model.joblib", results_dir=paths.RESULTS_DIR / "bow")
    model.load()
    assert model.available, model.load_error
    order = model.class_order
    samples = json.loads((paths.FRONTEND_DATA / "sampleCommands.json").read_text(encoding="utf-8"))
    texts = [t for cat in samples["categories"] for t in cat["commands"]] + ["", "zzqx unseen words"]
    cases = []
    sample_probabilities = {}

    def add(name, probs, thr):
        d = apply_threshold(probs, thr, order)
        cases.append({"name": name, "probabilities": probs, "threshold": thr,
                      "expected": {k: d[k] for k in ("raw_prediction", "confidence", "final_intent", "status",
                                                     "fallback_applied")} | {"action": d["application"]["action"]}})

    for text in texts:
        probs = model.predict(text).probabilities
        sample_probabilities[text] = probs
        conf = max(probs.values())
        for thr in GRID + [model.default_threshold]:
            add(f"{text!r} @ {thr}", probs, thr)
        # exact boundary: equal is accepted, one ulp above rejects
        add(f"{text!r} @ confidence (equal)", probs, conf)
        add(f"{text!r} @ confidence + 1ulp", probs, float(np.nextafter(conf, 1.0)))
        add(f"{text!r} @ confidence - 1ulp", probs, float(np.nextafter(conf, 0.0)))
    # ties: first class in class order wins (numpy argmax semantics)
    tie = {c: 0.0 for c in order}
    tie[order[2]] = tie[order[5]] = 0.5
    add("tie between two classes", tie, 0.5)
    uniform = {c: 1 / len(order) for c in order}
    add("uniform distribution", uniform, 0.1)
    add("uniform distribution, high threshold", uniform, 0.2)
    return {"generated_by": "demo/backend/generate_threshold_cases.py", "class_order": order,
            "default_threshold": model.default_threshold, "sample_probabilities": sample_probabilities,
            "cases": cases}


def build_workflow_cases(data: dict) -> list[dict]:
    """Every sample command (real probabilities) and every intent forced, at two thresholds."""
    order = data["class_order"]
    items = list(data["sample_probabilities"].items())
    items += [(t, {c: (0.9 if c == intent else 0.1 / (len(order) - 1)) for c in order})
              for t in EXTRA_WORKFLOW_TEXTS for intent in order]
    out = []
    for text, probs in items:
        ranked = [{"intent": k, "probability": v} for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)]
        for thr in (0.05, 0.95):
            d = apply_threshold(probs, thr, order)
            out.append({"text": text, "decision": d, "ranked": ranked, "expected": build_workflow(text, d, ranked)})
    return out


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    data = build_cases()
    FIXTURE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Wrote {len(data['cases'])} cases to {FIXTURE}")
    wf = build_workflow_cases(data)
    WORKFLOW_FIXTURE.write_text(json.dumps({"generated_by": "demo/backend/generate_threshold_cases.py", "cases": wf},
                                           indent=1), encoding="utf-8")
    print(f"Wrote {len(wf)} workflow cases to {WORKFLOW_FIXTURE}")


if __name__ == "__main__":
    main()
