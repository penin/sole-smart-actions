"""Model registry for the demo: the four primary models plus the historical Bag of Words baselines.

Every entry implements ``summary()``, ``info()`` and ``predict(text)`` and returns raw probabilities only;
thresholding is done separately by ``decision.apply_threshold``. Primary models are loaded through the
project's common interface (``sole_smart_actions.training.load_trained``) from ``artifacts/<model>/``;
large neural models load lazily on first use. All metrics shown come from the generated ``results/`` files.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import paths
from sole_intent.rejection import UNKNOWN_INTENT
from sole_intent.train import load_bundle, reorder_proba
from sole_intent.utils import sha256_file

FALLBACK_RULE = ("max probability < threshold -> UNKNOWN_INTENT -> app fallback FALLBACK_HELP_OR_CLARIFY "
                 "(offer Help & Support or ask to rephrase). Probability equal to the threshold is accepted. "
                 "A genuine HELP_SUPPORT prediction is routed to help, not treated as a fallback.")
DISPLAY = {"bow": "Bag of Words + Logistic Regression", "tfidf": "TF-IDF + Logistic Regression",
           "transformer": "Custom Transformer", "distilbert": "Fine-tuned DistilBERT"}
FAMILY = {"bow": "Word counts (unigrams + bigrams) -> multinomial Logistic Regression",
          "tfidf": "TF-IDF word + character n-grams -> Logistic Regression",
          "transformer": "Self-attention encoder trained from scratch (GloVe-initialised embeddings)",
          "distilbert": "Pretrained DistilBERT (66M parameters) fine-tuned on Sole commands"}


class ModelUnavailableError(RuntimeError):
    """Raised when a model is not trained yet or failed to load."""


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


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(paths.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).name


def unified_evaluation(results_dir: Path) -> tuple[dict, dict | None]:
    """Evaluation block from the standard result files written by ``scripts.evaluate``."""
    meta = _read_json(results_dir / "experiment_metadata.json")
    t = _read_json(results_dir / "metrics_test.json") or {}
    v = _read_json(results_dir / "metrics_validation.json") or {}
    lat = _read_json(results_dir / "latency_metrics.json") or {}
    default_row = {}
    sweep = results_dir / "threshold_sweep.csv"
    if sweep.exists():
        import csv
        with open(sweep, encoding="utf-8") as f:
            default_row = next((r for r in csv.DictReader(f) if r["threshold_label"] == "default"), {})
    fl = lambda k: float(default_row[k]) if default_row.get(k) not in (None, "") else None  # noqa: E731
    ev = {
        "source": f"{_rel(results_dir)} (held-out test split)",
        "test_n": t.get("n_samples"), "test_accuracy": t.get("accuracy"), "test_macro_f1": t.get("macro_f1"),
        "test_weighted_f1": t.get("weighted_f1"), "test_macro_precision": t.get("macro_precision"),
        "test_macro_recall": t.get("macro_recall"), "test_macro_roc_auc": t.get("macro_roc_auc_ovr"),
        "test_log_loss": t.get("log_loss"), "test_top2_accuracy": t.get("top2_accuracy"),
        "test_mean_confidence": t.get("mean_confidence"), "test_macro_f1_ci95": None,
        "validation_macro_f1": v.get("macro_f1"), "validation_accuracy": v.get("accuracy"),
        "expected_calibration_error": t.get("expected_calibration_error"),
        "known_intent_coverage_at_default": fl("known_coverage"),
        "unknown_rejection_rate_at_default": fl("unknown_rejection_rate"),
        "unknown_safe_fallback_at_default": fl("unknown_safe_fallback_rate"),
        "median_latency_ms": lat.get("p50_ms"), "p95_latency_ms": lat.get("p95_ms"),
        "training_seconds": (meta or {}).get("training_seconds_final_fit"),
        "model_size_mb": ((meta or {}).get("artifact") or {}).get("size_mb"),
        "n_parameters": (meta or {}).get("n_parameters"),
    }
    return ev, meta


class BowLogRegModel:
    """A Bag of Words + Logistic Regression bundle, loaded once, with exact per-word explanations.

    ``results_dir`` set: a primary-protocol model (metrics/threshold from results/bow/).
    Otherwise: a historical baseline (metrics from its original results/history/... folder)."""

    family = FAMILY["bow"]
    explainable = True

    def __init__(self, model_path: Path, metrics_dir: Path | None = None, model_id: str = "bow",
                 name: str = DISPLAY["bow"], results_dir: Path | None = None,
                 inference_config: Path = paths.INFERENCE_CONFIG, group: str = "primary"):
        self.id, self.name, self.group = model_id, name, group
        self.model_path = Path(model_path)
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None
        self.results_dir = Path(results_dir) if results_dir else None
        self.inference_config = Path(inference_config)
        self.available = False
        self.load_error: str | None = None
        self.bundle: dict | None = None

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
            self.default_threshold, self.threshold_source = self._resolve_default_threshold()
            self.model_sha256 = sha256_file(self.model_path)
            self.available, self.load_error = True, None
        except Exception as exc:  # surfaced through /api/health and 503 responses
            self.available, self.load_error = False, f"{type(exc).__name__}: {exc}"

    def _resolve_default_threshold(self) -> tuple[float, str]:
        if self.results_dir is not None:
            meta = _read_json(self.results_dir / "experiment_metadata.json")
            if not meta:
                raise FileNotFoundError(f"{_rel(self.results_dir)}/experiment_metadata.json missing; "
                                        "run python -m scripts.evaluate --model bow")
            return float(meta["threshold_calibration"]["threshold"]), \
                f"{_rel(self.results_dir)}/experiment_metadata.json (95% validation coverage)"
        try:
            cfg = yaml.safe_load(self.inference_config.read_text(encoding="utf-8")) or {}
            value = (cfg.get("inference") or {}).get("threshold")
        except OSError:
            value = None
        if value is not None:
            return float(value), _rel(self.inference_config)
        return float(self.bundle["rejection_threshold"]), "the saved model bundle"

    def _require(self) -> None:
        if not self.available:
            raise ModelUnavailableError(self.load_error or "model not loaded")

    def predict(self, text: str) -> RawPrediction:
        self._require()
        start = time.perf_counter()
        proba = reorder_proba(self.bundle["pipeline"], [text], self.class_order)[0]
        inference_ms = (time.perf_counter() - start) * 1000
        probabilities = {c: float(p) for c, p in zip(self.class_order, proba)}
        top = int(np.argmax(proba))
        return RawPrediction(model_id=self.id, text=text, class_order=list(self.class_order),
                             probabilities=probabilities, raw_prediction=self.class_order[top],
                             confidence=float(proba[top]), inference_ms=inference_ms,
                             explanation=self.explain(text, self.class_order[top]))

    def explain(self, text: str, predicted: str) -> dict:
        """Token recognition and exact per-feature contributions to the predicted class's score:
        ``intercept + sum(contributions) == decision_function`` for that class (verified in tests)."""
        self._require()
        normalized = self.normalizer.transform([text])[0]
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
        contributions = sorted(({"feature": str(self.feature_names[j]), "count": int(c), "weight": float(coef[k, j]),
                                 "contribution": float(c * coef[k, j]),
                                 "contribution_vs_runner_up": float(c * (coef[k, j] - coef[runner, j]))}
                                for j, c in zip(row.indices, row.data)), key=lambda c: c["contribution"], reverse=True)
        top_features = np.argsort(-coef[k])[:8]
        return {"kind": "bow_linear", "normalized_text": normalized, "tokens": tokens, "recognized_tokens": recognized,
                "unknown_tokens": unknown, "recognized_phrases": phrases, "ngram_range": list(self.vectorizer.ngram_range),
                "explained_class": predicted, "runner_up_class": self.classifier_classes[runner],
                "intercept": float(self.classifier.intercept_[k]), "class_score": float(scores[k]),
                "contributions": contributions,
                "top_class_features": [{"feature": str(self.feature_names[j]), "weight": float(coef[k, j])}
                                       for j in top_features]}

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "available": self.available, "group": self.group,
                "status": "available" if self.available else "error", "error": self.load_error}

    def _details(self, params: dict) -> list[dict]:
        ng = list(self.vectorizer.ngram_range)
        return [{"label": "Vectorizer", "value": f"CountVectorizer ({'unigrams' if ng == [1, 1] else f'{ng[0]}-{ng[1]}-grams'}"
                                                 f"{', binary' if self.vectorizer.binary else ''})"},
                {"label": "Classifier", "value": f"LogisticRegression, C = {params.get('C', self.classifier.C)}"},
                {"label": "Vocabulary size", "value": f"{len(self.vocabulary):,}"},
                {"label": "Parameters", "value": f"{int(self.classifier.coef_.size + self.classifier.intercept_.size):,}"}]

    def info(self) -> dict:
        self._require()
        base = {"id": self.id, "name": self.name, "group": self.group, "family": self.family, "explainable": True,
                "model_file": _rel(self.model_path), "model_sha256": self.model_sha256,
                "classes": list(self.class_order), "n_classes": len(self.class_order),
                "unknown_label": UNKNOWN_INTENT, "unknown_label_is_model_class": UNKNOWN_INTENT in self.class_order,
                "default_threshold": self.default_threshold, "default_threshold_source": self.threshold_source,
                "fallback_rule": FALLBACK_RULE}
        if self.results_dir is not None:
            ev, meta = unified_evaluation(self.results_dir)
            return {**base, "original_author": (meta or {}).get("original_author"),
                    "dataset_version": ((meta or {}).get("dataset") or {}).get("version"), "version": "primary",
                    "trained_on": (meta or {}).get("trained_on"),
                    "details": self._details((meta or {}).get("hyperparameters") or {}), "evaluation": ev}
        meta = self.bundle.get("metadata", {})
        m_test = _read_json(self.metrics_dir / "metrics_test.json") or {}
        m_val = _read_json(self.metrics_dir / "metrics_validation.json") or {}
        rej = (_read_json(self.metrics_dir / "rejection_metrics.json") or {}).get(
            "primary_frozen_threshold_test_plus_unknown", {})
        runtime = (_read_json(self.metrics_dir / "runtime_metrics.json") or {}).get("tuned", {})
        boot = (_read_json(self.metrics_dir / "bootstrap_ci_test.json") or {}).get("models", {}).get("tuned", {})
        return {**base, "original_author": "penin", "version": meta.get("model_version"),
                "dataset_version": (meta.get("dataset") or {}).get("dataset_version"), "trained_on": ["train"],
                "details": self._details(meta.get("params", {})),
                "evaluation": {
                    "source": f"{_rel(self.metrics_dir)} (historical run, held-out test split)",
                    "test_n": m_test.get("n_samples"), "test_accuracy": m_test.get("accuracy"),
                    "test_macro_f1": m_test.get("macro_f1"), "test_weighted_f1": m_test.get("weighted_f1"),
                    "test_macro_precision": m_test.get("macro_precision"), "test_macro_recall": m_test.get("macro_recall"),
                    "test_macro_roc_auc": m_test.get("macro_roc_auc_ovr"), "test_log_loss": m_test.get("log_loss"),
                    "test_top2_accuracy": m_test.get("top2_accuracy"), "test_mean_confidence": m_test.get("mean_confidence"),
                    "test_macro_f1_ci95": ([boot["macro_f1"]["ci_low"], boot["macro_f1"]["ci_high"]]
                                           if "macro_f1" in boot else None),
                    "validation_macro_f1": m_val.get("macro_f1"), "validation_accuracy": m_val.get("accuracy"),
                    "expected_calibration_error": m_test.get("expected_calibration_error"),
                    "known_intent_coverage_at_default": rej.get("known_intent_coverage"),
                    "unknown_rejection_rate_at_default": rej.get("unknown_rejection_rate"),
                    "unknown_safe_fallback_at_default": None,
                    "median_latency_ms": (runtime.get("latency_single_request") or {}).get("median_ms"),
                    "p95_latency_ms": (runtime.get("latency_single_request") or {}).get("p95_ms"),
                    "training_seconds": runtime.get("training_seconds"), "model_size_mb": runtime.get("model_size_mb"),
                    "n_parameters": int(self.classifier.coef_.size + self.classifier.intercept_.size)}}


class UnifiedModel:
    """TF-IDF, Custom Transformer or DistilBERT through the project's common model interface (lazy-loaded)."""

    explainable = False

    def __init__(self, model_name: str, artifacts_dir: Path | None = None, results_dir: Path | None = None,
                 lazy: bool = True):
        self.id = self.model_name = model_name
        self.name, self.family, self.group = DISPLAY[model_name], FAMILY[model_name], "primary"
        self.artifacts_dir = Path(artifacts_dir or paths.ARTIFACTS_DIR / model_name)
        self.results_dir = Path(results_dir or paths.RESULTS_DIR / model_name)
        self.model = None
        self.card: dict | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()
        self.meta = _read_json(self.results_dir / "experiment_metadata.json")
        if not (self.artifacts_dir / "model_card.json").exists():
            self.load_error = (f"Not trained yet: {_rel(self.artifacts_dir)} is missing. "
                               f"Run: python -m scripts.train --model {model_name}")
        elif self.meta is None:
            self.load_error = (f"Not evaluated yet: {_rel(self.results_dir)} is missing. "
                               f"Run: python -m scripts.evaluate --model {model_name}")
        if not lazy and self.load_error is None:
            self._ensure_loaded()

    @property
    def available(self) -> bool:
        return self.load_error is None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def class_order(self) -> list[str]:
        return list((self.meta or {}).get("class_order") or [])

    @property
    def default_threshold(self) -> float:
        return float(self.meta["threshold_calibration"]["threshold"])

    def _ensure_loaded(self):
        if self.load_error:
            raise ModelUnavailableError(self.load_error)
        if self.model is None:
            with self._lock:
                if self.model is None:
                    try:
                        from sole_smart_actions.training import load_trained
                        self.model, self.card = load_trained(self.model_name, self.artifacts_dir)
                    except Exception as exc:
                        self.load_error = f"{type(exc).__name__}: {exc}"
                        raise ModelUnavailableError(self.load_error) from exc
        return self.model

    def load(self) -> None:
        try:
            self._ensure_loaded()
        except ModelUnavailableError:
            pass

    def predict(self, text: str) -> RawPrediction:
        model = self._ensure_loaded()
        start = time.perf_counter()
        proba = model.predict_proba([text])[0]
        inference_ms = (time.perf_counter() - start) * 1000
        classes = model.class_order
        top = int(np.argmax(proba))
        return RawPrediction(model_id=self.id, text=text, class_order=list(classes),
                             probabilities={c: float(p) for c, p in zip(classes, proba)},
                             raw_prediction=classes[top], confidence=float(proba[top]), inference_ms=inference_ms,
                             explanation=self.explain(text))

    def explain(self, text: str) -> dict:
        """What the model actually sees: its own tokenization (no per-word attributions for these models)."""
        m = self.model
        base = {"normalized_text": text.strip(), "recognized_phrases": [], "contributions": [], "top_class_features": []}
        if self.model_name == "tfidf":
            word_vec = m.pipeline.named_steps["vec"]
            word_vec = dict(word_vec.transformer_list)["word"] if hasattr(word_vec, "transformer_list") else word_vec
            tokens = word_vec.build_tokenizer()(word_vec.build_preprocessor()(text.strip()))
            vocab = word_vec.vocabulary_
            return {**base, "kind": "tfidf", "tokens": tokens, "recognized_tokens": [t for t in tokens if t in vocab],
                    "unknown_tokens": [t for t in tokens if t not in vocab],
                    "description": "Words are weighted by TF-IDF; character n-grams (2-5 letters) also let the model "
                                   "match misspelled or unseen words, so unknown words still contribute partially."}
        if self.model_name == "transformer":
            from sole_smart_actions.models.transformer.vectorizer import standardize, tokenize
            tokens = tokenize(text)[: m.vectorizer.sequence_length]
            vocab = m.vectorizer.index
            return {**base, "kind": "transformer", "normalized_text": standardize(text), "tokens": tokens,
                    "recognized_tokens": [t for t in tokens if t in vocab],
                    "unknown_tokens": [t for t in tokens if t not in vocab],
                    "description": "Lowercased, punctuation removed (so '$' is dropped), up to 16 tokens. Unknown words "
                                   "become [UNK]. Self-attention lets each word's meaning depend on the others."}
        tokens = m.tokenizer.tokenize(text)[: m.cfg["text"]["max_tokens"] - 2]
        return {**base, "kind": "distilbert", "normalized_text": text.strip().lower(), "tokens": tokens,
                "recognized_tokens": [t for t in tokens if t != "[UNK]"],
                "unknown_tokens": [t for t in tokens if t == "[UNK]"],
                "description": "WordPiece splits rare words into known sub-words (e.g. 'jone' -> 'jo', '##ne'), so "
                               "almost nothing is out of vocabulary. Six pretrained attention layers read the whole sentence."}

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "available": self.available, "group": self.group,
                "status": "available" if self.available else "error", "error": self.load_error}

    def info(self) -> dict:
        if self.load_error:
            raise ModelUnavailableError(self.load_error)
        ev, meta = unified_evaluation(self.results_dir)
        hp = meta.get("hyperparameters") or {}
        details = [{"label": "Selected configuration", "value": ", ".join(f"{k}={v}" for k, v in hp.items())},
                   {"label": "Parameters", "value": f"{meta['n_parameters']:,}" if meta.get("n_parameters") else "n/a"},
                   {"label": "Model size", "value": f"{meta['artifact']['size_mb']:.1f} MB"}]
        if self.model_name == "distilbert":
            pre = (meta.get("selection") or {}).get("pretrained") or {}
            details.insert(0, {"label": "Pretrained", "value": f"{pre.get('name')}@{str(pre.get('revision'))[:7]}"})
        return {"id": self.id, "name": self.name, "group": self.group, "family": self.family, "explainable": False,
                "original_author": meta.get("original_author"), "version": "primary",
                "dataset_version": meta["dataset"]["version"], "trained_on": meta.get("trained_on"),
                "model_file": _rel(self.artifacts_dir),
                "model_sha256": meta["artifact"]["model_card_sha256"], "details": details,
                "classes": meta["class_order"], "n_classes": len(meta["class_order"]),
                "unknown_label": UNKNOWN_INTENT, "unknown_label_is_model_class": False,
                "default_threshold": self.default_threshold,
                "default_threshold_source": f"{_rel(self.results_dir)}/experiment_metadata.json (95% validation coverage)",
                "fallback_rule": FALLBACK_RULE, "loaded": self.loaded, "evaluation": ev}


class ModelRegistry:
    ALIASES = {"bow_logreg": "bow"}

    def __init__(self, models: list):
        self.models = {m.id: m for m in models}
        self.default_id = models[0].id

    def get(self, model_id: str | None):
        model_id = self.ALIASES.get(model_id, model_id) if model_id else self.default_id
        model = self.models.get(model_id)
        if model is None:
            raise KeyError(model_id)
        return model

    def primary(self) -> list:
        return [m for m in self.models.values() if m.group == "primary"]

    def summaries(self) -> list[dict]:
        return [m.summary() for m in self.models.values()]


def build_registry(*, preload: bool = False) -> ModelRegistry:
    bow = BowLogRegModel(paths.ARTIFACTS_DIR / "bow" / "model.joblib", model_id="bow", name=DISPLAY["bow"],
                         results_dir=paths.RESULTS_DIR / "bow")
    # The demo offers only the four current (Dataset V5) models; historical BoW runs stay in artifacts/history/.
    bow.load()
    others = [UnifiedModel(n, lazy=not preload) for n in ("tfidf", "transformer", "distilbert")]
    return ModelRegistry([bow, *others])
