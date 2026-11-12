"""The demo must use the existing trained model and reproduce the project's inference exactly."""

import json

import numpy as np
import pandas as pd
import pytest

import paths
from model_service import BowLogRegModel
from sole_intent.predict import predict_texts
from sole_intent.train import load_bundle

SAMPLES = json.loads((paths.FRONTEND_DATA / "sampleCommands.json").read_text(encoding="utf-8"))
SAMPLE_TEXTS = [t for c in SAMPLES["categories"] for t in c["commands"]]


def test_default_model_is_v5(model):
    assert model.model_path == paths.PROJECT_ROOT / "models_v5" / "bow_logreg_v5_tuned_v1.0.0.joblib"
    assert model.id == "bow_logreg"


def test_existing_model_and_vectorizer_load(model):
    assert type(model.vectorizer).__name__ == "CountVectorizer"
    assert type(model.classifier).__name__ == "LogisticRegression"
    assert type(model.normalizer).__name__ == "TextNormalizer"
    assert len(model.vocabulary) == len(load_bundle(model.model_path)["pipeline"].named_steps["bow"].vocabulary_)


def test_classes_come_from_loaded_classifier(model):
    assert sorted(model.class_order) == sorted(str(c) for c in model.classifier.classes_)
    assert len(model.class_order) == 9 and "UNKNOWN_INTENT" not in model.class_order


@pytest.mark.parametrize("which, expected", [("v5", 0.34152482), ("v4", 0.19572296)])
def test_default_threshold_matches_project_configuration(which, expected):
    from conftest import MODEL_SPECS
    m = BowLogRegModel(MODEL_SPECS[which][0], metrics_dir=MODEL_SPECS[which][1])
    m.load()
    assert m.default_threshold == load_bundle(m.model_path)["rejection_threshold"] == pytest.approx(expected, abs=1e-7)
    assert "bundle" in m.threshold_source  # config/inference.yaml has threshold: null


@pytest.mark.parametrize("text", SAMPLE_TEXTS + ["", "SEND $50 TO JONE!!!", "x" * 2000])
def test_predictions_match_existing_inference_pipeline(any_model, text):
    model = any_model
    expected = predict_texts([text], model.bundle)[0]
    got = model.predict(text)
    assert got.raw_prediction == expected["predicted_intent"]
    assert got.probabilities == expected["probabilities"]  # exact equality: same code path, same floats


def test_predictions_match_saved_test_predictions(any_model):
    """Probabilities for the held-out test rows equal those saved by that model's evaluation run."""
    model = any_model
    saved = pd.read_csv(model.results_dir / "predictions" / "test_predictions.csv")
    for _, row in saved.iloc[::9].iterrows():
        got = model.predict(row["text"]).probabilities
        np.testing.assert_allclose([got[c] for c in model.class_order],
                                   [row[f"prob_{c}"] for c in model.class_order], atol=1e-12)


def test_probabilities_are_mapped_to_labels(model):
    p = model.predict("Pay my EFL bill")
    assert list(p.probabilities) == model.class_order
    assert sum(p.probabilities.values()) == pytest.approx(1.0)
    native = model.bundle["pipeline"].predict_proba(["Pay my EFL bill"])[0]
    for i, cls in enumerate(model.classifier.classes_):
        assert p.probabilities[str(cls)] == pytest.approx(native[i], abs=0)


def test_preprocessing_is_the_trained_pipeline(model):
    e = model.predict("  SEND   $50 to Jóne ").explanation
    assert e["normalized_text"] == model.normalizer.transform(["  SEND   $50 to Jóne "])[0] == "send $50 to jóne"
    assert e["tokens"] == ["send", "$", "50", "to", "jóne"]
    assert "jóne" in e["unknown_tokens"] and "send" in e["recognized_tokens"]


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_feature_contributions_are_mathematically_exact(any_model, text):
    model = any_model
    p = model.predict(text)
    e = p.explanation
    X = model.vectorizer.transform([e["normalized_text"]])
    k = list(model.classifier.classes_).index(p.raw_prediction)
    total = e["intercept"] + sum(c["contribution"] for c in e["contributions"])
    assert total == pytest.approx(model.classifier.decision_function(X)[0, k], abs=1e-12)
    in_vocab = [f for f in model.analyzer(e["normalized_text"]) if f in model.vocabulary]
    assert sorted(in_vocab) == sorted(t for c in e["contributions"] for t in [c["feature"]] * c["count"])
    assert set(e["recognized_phrases"]) == {f for f in in_vocab if " " in f}


def test_model_load_failure_is_reported(tmp_path):
    m = BowLogRegModel(model_path=tmp_path / "missing.joblib")
    m.load()
    assert not m.available and "missing.joblib" in m.load_error


def test_demo_does_not_modify_artifacts(model, client, protected_hashes):
    from sole_intent.utils import sha256_file
    for t in SAMPLE_TEXTS:
        client.post("/api/predict", json={"text": t, "threshold": 0.9})
    client.post("/api/decide", json={"probabilities": model.predict("Send").probabilities, "threshold": 0.7})
    assert {p: sha256_file(p) for p in protected_hashes} == protected_hashes
