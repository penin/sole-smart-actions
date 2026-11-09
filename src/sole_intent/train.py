"""Pipeline construction, validation-based hyperparameter selection and model persistence.

Pipeline: raw text -> TextNormalizer -> CountVectorizer -> LogisticRegression -> probabilities.
Everything lives in one sklearn Pipeline so the vocabulary is only ever fitted on the data
passed to ``fit`` (the training split).
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .evaluate import multiclass_log_loss
from .preprocessing import TextNormalizer, build_vectorizer
from .utils import MODELS_DIR, load_config, save_json, set_seed, utc_now


def build_pipeline(params: dict, cfg: dict) -> Pipeline:
    pre = cfg["preprocessing"]
    seed = cfg["experiment"]["random_seed"]
    return Pipeline([
        ("normalize", TextNormalizer(unicode_form=pre["unicode_form"], lowercase=pre["lowercase"],
                                     collapse_whitespace=pre["collapse_whitespace"])),
        ("bow", build_vectorizer(ngram_range=params["ngram_range"], binary=params["binary"],
                                 token_pattern=pre["token_pattern"])),
        ("clf", LogisticRegression(C=params["C"], solver=cfg["model"]["solver"],
                                   max_iter=cfg["model"]["max_iter"], random_state=seed)),
    ])


def fit_pipeline(pipeline: Pipeline, texts, labels, class_order: list[str]) -> tuple[Pipeline, float]:
    """Fit and return (pipeline, seconds). Fails loudly on non-convergence or class mismatch."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(list(texts), list(labels))
    elapsed = time.perf_counter() - start
    if sorted(pipeline.classes_) != sorted(class_order):
        raise ValueError(f"Model classes {list(pipeline.classes_)} do not match {class_order}")
    return pipeline, elapsed


def reorder_proba(pipeline, texts, class_order: list[str]):
    """predict_proba with columns in the canonical class order (sklearn sorts alphabetically)."""
    proba = pipeline.predict_proba(list(texts))
    index = [list(pipeline.classes_).index(c) for c in class_order]
    return proba[:, index]


def param_grid(cfg: dict) -> list[dict]:
    grid = cfg["tuning"]["grid"]
    return [{"ngram_range": tuple(ng), "C": float(c), "binary": bool(b)}
            for ng, c, b in itertools.product(grid["ngram_range"], grid["C"], grid["binary"])]


def params_id(params: dict) -> str:
    ng = params["ngram_range"]
    return f"ngram{ng[0]}{ng[1]}_C{params['C']:g}_{'binary' if params['binary'] else 'count'}"


def tune_hyperparameters(train: pd.DataFrame, validation: pd.DataFrame, cfg: dict,
                         class_order: list[str]) -> tuple[pd.DataFrame, dict]:
    """Grid search scored on the validation split only. The test split is not an argument,
    so it cannot influence selection. Ties on macro F1 are broken by validation log loss."""
    records = []
    for params in param_grid(cfg):
        pipe, seconds = fit_pipeline(build_pipeline(params, cfg), train["text"], train["intent"], class_order)
        val_pred = pipe.predict(list(validation["text"]))
        val_proba = reorder_proba(pipe, validation["text"], class_order)
        tr_pred = pipe.predict(list(train["text"]))
        records.append({
            "params_id": params_id(params),
            "ngram_range": f"({params['ngram_range'][0]},{params['ngram_range'][1]})",
            "C": params["C"],
            "binary": params["binary"],
            "train_macro_f1": f1_score(train["intent"], tr_pred, labels=class_order, average="macro"),
            "validation_macro_f1": f1_score(validation["intent"], val_pred, labels=class_order, average="macro"),
            "validation_accuracy": float((val_pred == validation["intent"].to_numpy()).mean()),
            "validation_log_loss": multiclass_log_loss(validation["intent"], val_proba, class_order),
            "vocabulary_size": len(pipe.named_steps["bow"].vocabulary_),
            "fit_seconds": seconds,
        })
    results = pd.DataFrame(records).sort_values(
        ["validation_macro_f1", "validation_log_loss"], ascending=[False, True]).reset_index(drop=True)
    results.insert(0, "rank", range(1, len(results) + 1))
    best = results.iloc[0]
    ng = tuple(int(x) for x in best["ngram_range"].strip("()").split(","))
    best_params = {"ngram_range": ng, "C": float(best["C"]), "binary": bool(best["binary"])}
    return results, best_params


def save_bundle(pipeline: Pipeline, path: Path, *, class_order: list[str], threshold: float | None,
                metadata: dict) -> Path:
    """Persist pipeline + canonical class order + rejection threshold as one artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": pipeline, "class_order": list(class_order),
              "rejection_threshold": threshold, "metadata": metadata}
    joblib.dump(bundle, path)
    save_json(metadata | {"class_order": class_order, "rejection_threshold": threshold},
              path.with_suffix(".metadata.json"))
    return path


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    for key in ("pipeline", "class_order", "rejection_threshold"):
        if key not in bundle:
            raise ValueError(f"Model bundle {path} missing '{key}'")
    return bundle


def main() -> None:
    """Train-only entry point: tune on validation and save the selected model (no evaluation)."""
    from .data_loader import get_splits, load_dataset, validate_schema
    from .rejection import max_confidence, threshold_for_coverage

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["random_seed"])
    df = load_dataset(cfg)
    validate_schema(df, cfg)
    splits = get_splits(df, cfg)
    labels = cfg["labels"]
    results, best = tune_hyperparameters(splits.train, splits.validation, cfg, labels)
    print(results.to_string(index=False))
    pipe, seconds = fit_pipeline(build_pipeline(best, cfg), splits.train["text"], splits.train["intent"], labels)
    thr = threshold_for_coverage(max_confidence(reorder_proba(pipe, splits.validation["text"], labels)),
                                 cfg["rejection"]["target_known_coverage"])
    exp = cfg["experiment"]
    path = MODELS_DIR / f"{exp['experiment_id']}_tuned_v{exp['model_version']}.joblib"
    save_bundle(pipe, path, class_order=labels, threshold=thr,
                metadata={"params": best, "training_seconds": seconds, "created_utc": utc_now()})
    print(f"Saved {path} (threshold={thr:.4f})")


if __name__ == "__main__":
    main()
