"""Generate the shared threshold-decision fixture used by BOTH test suites.

The backend's ``decision.apply_threshold`` (wrapping the project's ``apply_rejection``) produces the expected
results; the frontend's ``applyThreshold`` must reproduce them exactly (``src/__tests__/threshold.test.ts``).
Probabilities come from the real model on every sample command, plus exact-boundary and tie cases.

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
from model_service import BowLogRegModel  # noqa: E402

FIXTURE = paths.DEMO_DIR / "frontend" / "src" / "__tests__" / "fixtures" / "threshold_cases.json"
GRID = [0.0, 0.01, 0.1, 0.25, 0.3, 0.35, 0.5, 0.6, 0.8, 0.99, 1.0]


def build_cases() -> dict:
    model = BowLogRegModel()
    model.load()
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


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    data = build_cases()
    FIXTURE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Wrote {len(data['cases'])} cases to {FIXTURE}")


if __name__ == "__main__":
    main()
