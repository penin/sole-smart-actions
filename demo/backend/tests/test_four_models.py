"""The four-model demo: TF-IDF / Transformer / DistilBERT adapters, compare endpoints and the simulated workflow."""

import warnings

import numpy as np
import pandas as pd
import pytest

import paths
from conftest import NEURAL_AND_TFIDF, trained
from model_service import ModelRegistry, UnifiedModel
from workflow import FINANCIAL, build_workflow, extract_entities

needs = {m: pytest.mark.skipif(not trained(m), reason=f"{m} not trained/evaluated yet") for m in NEURAL_AND_TFIDF}


@pytest.mark.parametrize("name", [pytest.param(m, marks=needs[m]) for m in NEURAL_AND_TFIDF])
def test_unified_model_reproduces_saved_evaluation(registry, name):
    """Demo probabilities equal those written by scripts.evaluate for the held-out test rows."""
    m = registry.get(name)
    saved = pd.read_csv(paths.RESULTS_DIR / name / "test_predictions.csv")
    rows = saved.iloc[::15]
    got = np.array([[m.predict(t).probabilities[c] for c in m.class_order] for t in rows["text"]])
    want = rows[[f"prob_{c}" for c in m.class_order]].to_numpy()
    # Single-command inference vs batched evaluation: identical for sklearn; float32 GPU kernels differ slightly.
    np.testing.assert_allclose(got, want, atol=1e-9 if name == "tfidf" else 2e-4)
    assert (got.argmax(1) == want.argmax(1)).all()


@pytest.mark.parametrize("name", [pytest.param(m, marks=needs[m]) for m in NEURAL_AND_TFIDF])
def test_unified_model_info_uses_generated_results(client, name):
    info = client.get(f"/api/model-info?model_id={name}").json()
    import json
    meta = json.loads((paths.RESULTS_DIR / name / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert info["evaluation"]["test_accuracy"] == meta["headline"]["raw_test_accuracy"]
    assert info["default_threshold"] == meta["threshold_calibration"]["threshold"]
    assert info["classes"] == meta["class_order"] and info["n_classes"] == 9
    assert info["unknown_label_is_model_class"] is False


def test_untrained_model_is_reported_not_faked(tmp_path):
    from fastapi.testclient import TestClient

    from main import create_app
    m = UnifiedModel("distilbert", artifacts_dir=tmp_path / "none", results_dir=tmp_path / "none")
    assert not m.available and "python -m scripts.train --model distilbert" in m.load_error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with TestClient(create_app(ModelRegistry([m]))) as c:
            r = c.post("/api/predict", json={"text": "Send $50 to Jone", "model_id": "distilbert"})
            assert r.status_code == 503 and "not trained" in r.json()["detail"].lower()
            entry = c.post("/api/compare", json={"text": "Send $50 to Jone"}).json()[0]
            assert entry["ok"] is False and entry["prediction"] is None


def test_compare_runs_every_primary_model(client, registry):
    out = client.post("/api/compare", json={"text": "Pay my EFL bill"}).json()
    assert [e["model_id"] for e in out] == ["bow", "tfidf", "transformer", "distilbert"]
    for e in out:
        assert e["ok"] is trained(e["model_id"]) or e["model_id"] == "bow"
        if e["ok"]:
            p = e["prediction"]
            assert p["threshold"] == registry.get(e["model_id"]).default_threshold  # each model's own default
            assert abs(sum(p["probabilities"].values()) - 1) < 1e-6 and len(p["ranked"]) == 9
            assert p["workflow"]["kind"] in {"route", "confirm", "clarify"}
    shared = client.post("/api/compare", json={"text": "Pay my EFL bill", "threshold": 0.99}).json()
    assert all(e["prediction"]["threshold"] == 0.99 for e in shared if e["ok"])


def test_comparison_endpoint_serves_generated_file(client):
    r = client.get("/api/comparison")
    if not (paths.COMPARISON_DIR / "model_comparison.json").exists():
        assert r.status_code == 404
        return
    data = r.json()
    rows = {m["model"]: m for m in data["models"]}
    import json
    for name, row in rows.items():
        meta = json.loads((paths.RESULTS_DIR / name / "experiment_metadata.json").read_text(encoding="utf-8"))
        assert row["raw_test_accuracy"] == meta["headline"]["raw_test_accuracy"]  # not hard-coded


# ------------------------------------------------------------------ simulated workflow (no money ever moves)
def _decision(final, conf=0.9, thr=0.3):
    return {"raw_prediction": final if final != "UNKNOWN_INTENT" else "TRANSFER_MONEY", "confidence": conf,
            "threshold": thr, "final_intent": final}


RANKED = [{"intent": "TRANSFER_MONEY"}, {"intent": "BILL_PAY"}, {"intent": "HELP_SUPPORT"}]


def test_implicit_money_request_becomes_a_question():
    w = build_workflow("I owe Mary $50", _decision("TRANSFER_MONEY"), RANKED)
    assert w["message"] == "Would you like to send $50 to Mary?"
    assert w["kind"] == "confirm" and w["requires_confirmation"] is True


@pytest.mark.parametrize("intent", sorted(FINANCIAL))
def test_money_moving_intents_always_require_confirmation(intent):
    w = build_workflow("Send $5 to Jone", _decision(intent), RANKED)
    assert w["requires_confirmation"] is True and "never executes" in w["note"]
    assert "pin" not in w["message"].lower()


def test_low_confidence_asks_for_clarification_instead_of_acting():
    w = build_workflow("Payment", _decision("UNKNOWN_INTENT", conf=0.2, thr=0.34), RANKED)
    assert w["kind"] == "clarify" and w["screen"] is None and w["requires_confirmation"] is False
    assert w["options"][:3] == ["TRANSFER_MONEY", "BILL_PAY", "HELP_SUPPORT"]


def test_entity_extraction_is_conservative():
    assert extract_entities("Send $50 to Jone") == {"amount": "50", "recipient": "Jone", "biller": None}
    assert extract_entities("Pay $75 to EFL") == {"amount": "75", "recipient": None, "biller": "EFL"}
    assert extract_entities("Show my savings group")["amount"] is None
