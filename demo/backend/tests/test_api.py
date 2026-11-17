"""HTTP API behaviour."""

import json

import pytest

import paths
from conftest import trained
from schemas import MAX_TEXT_LENGTH

SAMPLES = json.loads((paths.FRONTEND_DATA / "sampleCommands.json").read_text(encoding="utf-8"))


def test_health_and_models(client):
    assert client.get("/api/health").json()["model_loaded"] is True
    listed = client.get("/api/models").json()
    assert [m["id"] for m in listed] == ["bow", "tfidf", "transformer", "distilbert"]  # current models only
    models = {m["id"]: m for m in listed}
    assert models["bow"]["available"] is True and models["bow"]["group"] == "primary"
    assert client.get("/api/model-info", params={"model_id": "bow_v4_history"}).status_code == 404
    for mid in ("tfidf", "transformer", "distilbert"):
        assert models[mid]["available"] is trained(mid)
        if not trained(mid):
            assert "python -m scripts" in models[mid]["error"]


def test_model_info_reads_real_artifacts(client, model):
    info = client.get("/api/model-info").json()
    metrics = json.loads((paths.RESULTS_DIR / "bow" / "metrics_test.json").read_text(encoding="utf-8"))
    assert info["classes"] == model.class_order and info["n_classes"] == 9
    details = {d["label"]: d["value"] for d in info["details"]}
    assert details["Vocabulary size"] == f"{len(model.vocabulary):,}" and "CountVectorizer" in details["Vectorizer"]
    assert info["default_threshold"] == model.default_threshold
    assert info["evaluation"]["test_accuracy"] == metrics["accuracy"]
    assert info["evaluation"]["test_macro_f1"] == metrics["macro_f1"]
    assert info["evaluation"]["test_weighted_f1"] == metrics["weighted_f1"]
    assert info["unknown_label_is_model_class"] is False


def test_predict_returns_full_real_distribution(client, model):
    r = client.post("/api/predict", json={"text": "Send $50 to Jone", "threshold": 0.3}).json()
    expected = model.predict("Send $50 to Jone")
    assert r["probabilities"] == expected.probabilities
    assert set(r["probabilities"]) == set(model.class_order)
    assert [x["probability"] for x in r["ranked"]] == sorted(r["probabilities"].values(), reverse=True)
    assert r["raw_prediction"] == expected.raw_prediction and r["confidence"] == expected.confidence
    assert r["threshold"] == 0.3 and r["recognized_tokens"] and r["inference_ms"] >= 0


def test_predict_uses_default_threshold_when_omitted(client, model):
    assert client.post("/api/predict", json={"text": "Help"}).json()["threshold"] == model.default_threshold


def test_threshold_changes_decision_not_probabilities(client):
    lo = client.post("/api/predict", json={"text": "Payment", "threshold": 0.0}).json()
    hi = client.post("/api/predict", json={"text": "Payment", "threshold": 1.0}).json()
    assert lo["probabilities"] == hi["probabilities"] and lo["raw_prediction"] == hi["raw_prediction"]
    assert lo["final_intent"] == lo["raw_prediction"] and hi["final_intent"] == "UNKNOWN_INTENT"
    assert hi["application"]["action"] == "FALLBACK_HELP_OR_CLARIFY"


def test_decide_endpoint_recomputes_without_inference(client):
    p = client.post("/api/predict", json={"text": "Show my savings group"}).json()
    for t in (0.0, p["confidence"], 1.0):
        d = client.post("/api/decide", json={"probabilities": p["probabilities"], "threshold": t}).json()
        assert d["raw_prediction"] == p["raw_prediction"]
        assert d["final_intent"] == (p["raw_prediction"] if t <= p["confidence"] else "UNKNOWN_INTENT")
    bad = client.post("/api/decide", json={"probabilities": {"X": 1.0}, "threshold": 0.5})
    assert bad.status_code == 422


@pytest.mark.parametrize("cat", SAMPLES["categories"], ids=lambda c: c["id"])
def test_every_sample_command_can_be_classified(client, cat):
    for text in cat["commands"]:
        r = client.post("/api/predict", json={"text": text})
        assert r.status_code == 200, text
        assert r.json()["final_intent"] in r.json()["class_order"] + ["UNKNOWN_INTENT"]


@pytest.mark.parametrize("payload, status", [
    ({"text": ""}, 422), ({"text": "   "}, 422), ({}, 422),
    ({"text": "x" * (MAX_TEXT_LENGTH + 1)}, 422),
    ({"text": "ok", "threshold": 1.5}, 422), ({"text": "ok", "threshold": -0.1}, 422),
    ({"text": "ok", "model_id": "nope"}, 404),
])
def test_malformed_requests(client, payload, status):
    r = client.post("/api/predict", json=payload)
    assert r.status_code == status
    assert "detail" in r.json()
    assert payload.get("text", "") not in json.dumps(r.json()) or payload.get("text", "") in ("", "ok")


def test_long_input_within_limit(client):
    r = client.post("/api/predict", json={"text": "send money " * 180})
    assert r.status_code == 200 and r.json()["raw_prediction"]


def test_model_load_failure_returns_503(tmp_path):
    import warnings

    from main import create_app
    from model_service import BowLogRegModel, ModelRegistry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from fastapi.testclient import TestClient
    broken = BowLogRegModel(model_path=tmp_path / "gone.joblib")
    broken.load()
    with TestClient(create_app(ModelRegistry([broken]))) as c:
        assert c.get("/api/health").json()["model_loaded"] is False
        r = c.post("/api/predict", json={"text": "Send $50 to Jone"})
        assert r.status_code == 503 and "gone.joblib" in r.json()["detail"]
