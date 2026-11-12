"""Model registry and the Bag-of-Words + Logistic Regression inference service.

Every model implements the same small interface (``info()`` and ``predict(text)``) and returns raw
probabilities only; thresholding is done separately by ``decision.apply_threshold``. Future models
(TF-IDF, custom Transformer, DistilBERT) plug in by adding a class and a registry entry.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

import paths
from sole_intent.rejection import UNKNOWN_INTENT
from sole_intent.train import load_bundle, reorder_proba
from sole_intent.utils import sha256_file


class ModelUnavailableError(RuntimeError):
    """Raised when a model is not implemented yet or failed to load."""


@dataclass
class RawPrediction:
    model_id: str
    text: str
    class_order: list[str]
    probabilities: dict[str, float]
    raw_prediction: str
    confidence: float
    inference_ms: float
    explanation: dict[str, Any] = field(default_factory=dict)


class IntentModel(Protocol):
    id: str
    name: str
    available: bool

    def summary(self) -> dict: ...
    def info(self) -> dict: ...
    def predict(self, text: str) -> RawPrediction: ...


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class BowLogRegModel:
    """A trained Bag of Words model from the project (V5 by default), loaded once, read-only."""

    family = "Bag of Words (CountVectorizer) + multinomial Logistic Regression"

    def __init__(self, model_path: Path = paths.MODEL_PATH, inference_config: Path = paths.INFERENCE_CONFIG,
                 metrics_dir: Path = paths.METRICS_DIR, model_id: str = "bow_logreg",
                 name: str = "Bag of Words + Logistic Regression (Dataset V5)"):
        self.id = model_id
        self.name = name
        self.model_path = Path(model_path)
        self.inference_config = Path(inference_config)
        self.metrics_dir = Path(metrics_dir)
        self.available = False
        self.load_error: str | None = None
        self.bundle: dict | None = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        try:
            self.bundle = load_bundle(self.model_path)
            pipe = self.bundle["pipeline"]
            self.normalizer = pipe.named_steps["normalize"]
            self.vectorizer = pipe.named_steps["bow"]
            self.classifier = pipe.named_steps["clf"]
            self.class_order = [str(c) for c in self.bundle["class_order"]]
            classifier_classes = [str(c) for c in self.classifier.classes_]
            if sorted(classifier_classes) != sorted(self.class_order):
                raise ValueError(f"bundle class order {self.class_order} != classifier classes {classifier_classes}")
            self.classifier_classes = classifier_classes
            self.vocabulary = self.vectorizer.vocabulary_
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.analyzer = self.vectorizer.build_analyzer()
            self.bundle_threshold = float(self.bundle["rejection_threshold"])
            self.default_threshold, self.threshold_source = self._resolve_default_threshold()
            self.model_sha256 = sha256_file(self.model_path)
            self.available = True
            self.load_error = None
        except Exception as exc:  # surfaced through /api/health and 503 responses
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _resolve_default_threshold(self) -> tuple[float, str]:
        """Same precedence as the project's predict CLI (minus --threshold): config value if set,
        otherwise the threshold stored in the model bundle."""
        try:
            cfg = yaml.safe_load(self.inference_config.read_text(encoding="utf-8")) or {}
            value = (cfg.get("inference") or {}).get("threshold")
        except OSError:
            value = None
        if value is not None:
            return float(value), f"config/{self.inference_config.name}"
        return self.bundle_threshold, "the saved model bundle"

    def _require(self) -> None:
        if not self.available:
            raise ModelUnavailableError(self.load_error or "model not loaded")

    # ------------------------------------------------------------------ inference
    def predict(self, text: str) -> RawPrediction:
        self._require()
        start = time.perf_counter()
        proba = reorder_proba(self.bundle["pipeline"], [text], self.class_order)[0]
        inference_ms = (time.perf_counter() - start) * 1000
        probabilities = {c: float(p) for c, p in zip(self.class_order, proba)}
        top = int(np.argmax(proba))
        return RawPrediction(
            model_id=self.id, text=text, class_order=list(self.class_order), probabilities=probabilities,
            raw_prediction=self.class_order[top], confidence=float(proba[top]), inference_ms=inference_ms,
            explanation=self.explain(text, self.class_order[top]),
        )

    def explain(self, text: str, predicted: str) -> dict:
        """Token recognition and exact per-feature contributions to the predicted class's score.

        For multinomial LR the class score is ``logit_k = intercept_k + Σ_j count_j · coef_kj`` and
        ``P(k) = softmax(logit)_k``. Contributions below are the ``count_j · coef_kj`` terms, so
        ``intercept + Σ contributions`` equals ``decision_function`` for class k (verified in tests)."""
        self._require()
        normalized = self.normalizer.transform([text])[0]
        # Single words (what a customer typed) vs n-gram features (what the model counts; e.g. bigrams for V5).
        tokens = self.vectorizer.build_tokenizer()(self.vectorizer.build_preprocessor()(normalized))
        recognized = [t for t in tokens if t in self.vocabulary]
        unknown = [t for t in tokens if t not in self.vocabulary]
        phrases = [f for f in self.analyzer(normalized) if " " in f and f in self.vocabulary]
        X = self.vectorizer.transform([normalized])
        k = self.classifier_classes.index(predicted)
        coef = self.classifier.coef_
        scores = self.classifier.decision_function(X)[0]
        order = np.argsort(-scores)
        runner = int(order[1]) if order[0] == k else int(order[0])
        row = X.tocsr()
        contributions = []
        for j, count in zip(row.indices, row.data):
            contributions.append({
                "feature": str(self.feature_names[j]),
                "count": int(count),
                "weight": float(coef[k, j]),
                "contribution": float(count * coef[k, j]),
                "contribution_vs_runner_up": float(count * (coef[k, j] - coef[runner, j])),
            })
        contributions.sort(key=lambda c: c["contribution"], reverse=True)
        top_features = np.argsort(-coef[k])[:8]
        return {
            "normalized_text": normalized,
            "tokens": tokens,
            "recognized_tokens": recognized,
            "unknown_tokens": unknown,
            "recognized_phrases": phrases,
            "ngram_range": list(self.vectorizer.ngram_range),
            "explained_class": predicted,
            "runner_up_class": self.classifier_classes[runner],
            "intercept": float(self.classifier.intercept_[k]),
            "class_score": float(scores[k]),
            "contributions": contributions,
            "top_class_features": [{"feature": str(self.feature_names[j]), "weight": float(coef[k, j])}
                                   for j in top_features],
        }

    # ------------------------------------------------------------------ metadata
    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "available": self.available,
                "status": "available" if self.available else "error", "error": self.load_error}

    def info(self) -> dict:
        self._require()
        meta = self.bundle.get("metadata", {})
        m_test = _read_json(self.metrics_dir / "metrics_test.json") or {}
        m_val = _read_json(self.metrics_dir / "metrics_validation.json") or {}
        rej = _read_json(self.metrics_dir / "rejection_metrics.json") or {}
        runtime = (_read_json(self.metrics_dir / "runtime_metrics.json") or {}).get("tuned", {})
        boot = ((_read_json(self.metrics_dir / "bootstrap_ci_test.json") or {}).get("models", {}).get("tuned", {}))
        primary = rej.get("primary_frozen_threshold_test_plus_unknown", {})
        params = meta.get("params", {})
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "version": meta.get("model_version"),
            "experiment_id": meta.get("experiment_id"),
            "dataset_version": (meta.get("dataset") or {}).get("dataset_version"),
            "model_file": str(self.model_path.relative_to(paths.PROJECT_ROOT)).replace("\\", "/"),
            "model_sha256": self.model_sha256,
            "pipeline_steps": [f"{name}: {type(step).__name__}" for name, step in self.bundle["pipeline"].steps],
            "vectorizer": {
                "type": type(self.vectorizer).__name__,
                "ngram_range": list(self.vectorizer.ngram_range),
                "binary": bool(self.vectorizer.binary),
                "token_pattern": self.vectorizer.token_pattern,
                "vocabulary_size": len(self.vocabulary),
            },
            "classifier": {"type": type(self.classifier).__name__, "C": params.get("C", self.classifier.C),
                           "n_parameters": int(self.classifier.coef_.size + self.classifier.intercept_.size)},
            "classes": list(self.class_order),
            "n_classes": len(self.class_order),
            "unknown_label": UNKNOWN_INTENT,
            "unknown_label_is_model_class": UNKNOWN_INTENT in self.class_order,
            "default_threshold": self.default_threshold,
            "default_threshold_source": self.threshold_source,
            "fallback_rule": ("max probability < threshold → UNKNOWN_INTENT → app fallback "
                              "FALLBACK_HELP_OR_CLARIFY (offer Help & Support or ask to rephrase). "
                              "Probability equal to the threshold is accepted. A genuine HELP_SUPPORT "
                              "prediction is routed to help, not treated as a fallback."),
            "evaluation": {
                "source": f"{self.metrics_dir.relative_to(paths.PROJECT_ROOT).as_posix()} (held-out test split)",
                "test_n": m_test.get("n_samples"),
                "test_accuracy": m_test.get("accuracy"),
                "test_macro_f1": m_test.get("macro_f1"),
                "test_weighted_f1": m_test.get("weighted_f1"),
                "test_macro_precision": m_test.get("macro_precision"),
                "test_macro_recall": m_test.get("macro_recall"),
                "test_macro_roc_auc": m_test.get("macro_roc_auc_ovr"),
                "test_log_loss": m_test.get("log_loss"),
                "test_top2_accuracy": m_test.get("top2_accuracy"),
                "test_mean_confidence": m_test.get("mean_confidence"),
                "test_macro_f1_ci95": ([boot["macro_f1"]["ci_low"], boot["macro_f1"]["ci_high"]]
                                       if "macro_f1" in boot else None),
                "validation_macro_f1": m_val.get("macro_f1"),
                "expected_calibration_error": m_test.get("expected_calibration_error"),
                "known_intent_coverage_at_default": primary.get("known_intent_coverage"),
                "unknown_rejection_rate_at_default": primary.get("unknown_rejection_rate"),
                "median_latency_ms": (runtime.get("latency_single_request") or {}).get("median_ms"),
                "training_seconds": runtime.get("training_seconds"),
                "model_size_mb": runtime.get("model_size_mb"),
            },
        }


class PlannedModel:
    """Placeholder for a classifier that has not been built yet. It never produces predictions."""

    def __init__(self, model_id: str, name: str):
        self.id, self.name, self.available = model_id, name, False

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "available": False, "status": "coming_soon", "error": None}

    def info(self) -> dict:
        raise ModelUnavailableError(f"{self.name} is not implemented yet")

    def predict(self, text: str) -> RawPrediction:
        raise ModelUnavailableError(f"{self.name} is not implemented yet")


class ModelRegistry:
    def __init__(self, models: list):
        self.models = {m.id: m for m in models}
        self.default_id = models[0].id

    def get(self, model_id: str | None):
        model = self.models.get(model_id or self.default_id)
        if model is None:
            raise KeyError(model_id)
        return model

    def summaries(self) -> list[dict]:
        return [m.summary() for m in self.models.values()]


def build_registry() -> ModelRegistry:
    v5 = BowLogRegModel()  # default: retrained on Dataset V5
    v4 = BowLogRegModel(paths.MODEL_PATH_V4, metrics_dir=paths.METRICS_DIR_V4, model_id="bow_logreg_v4",
                        name="Bag of Words + Logistic Regression (Dataset V4)")
    for m in (v5, v4):
        m.load()
    return ModelRegistry([
        v5,
        v4,
        PlannedModel("tfidf_logreg", "TF-IDF + Logistic Regression"),
        PlannedModel("custom_transformer", "Custom Transformer"),
        PlannedModel("distilbert", "DistilBERT"),
    ])
