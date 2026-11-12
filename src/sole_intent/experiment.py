"""End-to-end experiment: audit -> tune -> fit -> evaluate -> reject -> analyse -> report.

Invoked by ``scripts/run_all.py``. Every output is overwritten on each run (idempotent).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, learning_curve

from . import plots
from .audit import add_audit_keys, audit_dataset, leakage_table, strict_dataset_checks
from .data_loader import DatasetError, dataset_fingerprint, get_splits, load_dataset, validate_schema
from .evaluate import (cluster_bootstrap, confusion_frames, evaluate_predictions, expected_calibration_error,
                       measure_latency, per_class_frame, summary_row)
from .preprocessing import canonical_key, normalize_text, tokenize
from .rejection import (UNKNOWN_INTENT, apply_rejection, balanced_threshold, max_confidence,
                        rejection_outcomes, threshold_for_coverage, threshold_sweep)
from .reports import write_reports
from .train import build_pipeline, fit_pipeline, params_id, reorder_proba, save_bundle, tune_hyperparameters
from .utils import (DEFAULT_CONFIG, OutputPaths, environment_info, load_config, load_json, output_paths,
                    package_versions, resolve_path, save_json, set_seed, sha256_file, utc_now)

MONEY_MOVING = ["TRANSFER_MONEY", "BILL_PAY", "WITHDRAW_MONEY"]

# Qualitative diagnostic probes written for this project (not sampled from the dataset).
# "expected" is the author's judgement; AMBIGUOUS marks commands with no single correct intent.
DIAGNOSTIC_PROBES = [
    ("send fifty dollars to Jone", "TRANSFER_MONEY", "number written as words"),
    ("transfer $20 to Mere", "TRANSFER_MONEY", "currency symbol"),
    ("SEND $500 TO JONE!!!", "TRANSFER_MONEY", "capitalisation and punctuation"),
    ("snd 50 2 jone", "TRANSFER_MONEY", "SMS-style spelling"),
    ("Bula, send $30 to Jone vinaka", "TRANSFER_MONEY", "Fijian greeting/thanks code-switching"),
    ("pay Jone back $40", "TRANSFER_MONEY", "'pay' + person: TRANSFER vs BILL_PAY risk"),
    ("pay my power bill", "BILL_PAY", "synonym for electricity"),
    ("pay the water authority bill", "BILL_PAY", "utility name variant"),
    ("show my last five transactions", "VIEW_TRANSACTIONS", "number as word"),
    ("Jone sent me $50 did it arrive", "VIEW_TRANSACTIONS", "incoming transfer; shares words with TRANSFER"),
    ("what's my balance", "CHECK_BALANCE", "contraction"),
    ("how much is left in my wallet", "CHECK_BALANCE", "paraphrase"),
    ("how much did I spend this week", "CHECK_SPENDING", "stretch intent"),
    ("what did I spend the most on last month", "CHECK_SPENDING", "stretch intent paraphrase"),
    ("cash out 100 dollars at the agent", "WITHDRAW_MONEY", "agent cash-out phrasing"),
    ("add Mere to my contacts", "MANAGE_CONTACTS", "prompt example"),
    ("remove Jone from my saved payees", "MANAGE_CONTACTS", "payee vocabulary overlaps BILL_PAY"),
    ("show me my family savings group", "VIEW_GROUP", "group paraphrase"),
    ("who has paid into the church group", "VIEW_GROUP", "'paid' overlaps BILL_PAY"),
    ("I think someone stole my pin", "HELP_SUPPORT", "security issue"),
    ("my transfer didn't go through", "HELP_SUPPORT", "'transfer' keyword but support intent"),
    ("how do I send money", "HELP_SUPPORT", "how-to question; debatable vs TRANSFER_MONEY"),
    ("tell me a joke", UNKNOWN_INTENT, "out of scope"),
    ("who won the rugby", UNKNOWN_INTENT, "out of scope, local topic"),
    ("what is the weather in Suva", UNKNOWN_INTENT, "out of scope"),
    ("book a flight to Nadi", UNKNOWN_INTENT, "out of scope, transactional-sounding"),
    ("send me a pizza", UNKNOWN_INTENT, "out of scope but uses 'send'"),
    ("send money and pay my bill", "AMBIGUOUS", "multi-intent command"),
    ("don't send money to Jone", "AMBIGUOUS", "negation"),
]

# Pairs with (nearly) the same bag of words but different meaning.
WORD_ORDER_PROBES = [
    ("send money to jone", "jone send money to"),
    ("jone sent me money", "i sent jone money"),
    ("pay bill then check balance", "check balance then pay bill"),
    ("transfer from mere to jone", "transfer from jone to mere"),
    ("did i send money to mere", "i did send money to mere"),
]


def _prediction_frame(df: pd.DataFrame, proba: np.ndarray, class_order: list[str], threshold: float,
                      actual_col: str = "intent") -> pd.DataFrame:
    order = np.argsort(-proba, axis=1)
    labels = np.asarray(class_order, dtype=object)
    pred = labels[order[:, 0]]
    conf = proba.max(axis=1)
    second_conf = np.take_along_axis(proba, order[:, 1:2], axis=1).ravel()
    final = apply_rejection(proba, class_order, threshold)
    out = pd.DataFrame({
        "row_id": df["row_id"].to_numpy(), "excel_row": df["excel_row"].to_numpy(),
        "text": df["text"].to_numpy(), "split": df["split"].to_numpy(), "scope": df["scope"].to_numpy(),
        "actual_intent": df[actual_col].to_numpy(), "predicted_intent": pred, "final_intent": final,
        "max_confidence": conf, "second_intent": labels[order[:, 1]], "top2_margin": conf - second_conf,
    })
    out["correct"] = out["predicted_intent"] == out["actual_intent"]
    out["rejected"] = out["final_intent"] == UNKNOWN_INTENT
    out["final_correct"] = out["final_intent"] == out["actual_intent"]
    for i, c in enumerate(class_order):
        out[f"prob_{c}"] = proba[:, i]
    return out


def _oov_info(texts, vocabulary: set[str], token_pattern: str) -> tuple[list[list[str]], list[float]]:
    tokens = [tokenize(normalize_text(t), token_pattern) for t in texts]
    oov = [[w for w in toks if w not in vocabulary] for toks in tokens]
    frac = [len(o) / len(t) if t else np.nan for o, t in zip(oov, tokens)]
    return oov, frac


def _top_features(pipe, class_order: list[str], k: int) -> pd.DataFrame:
    vocab = pipe.named_steps["bow"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    rows = []
    for label in class_order:
        coefs = clf.coef_[list(clf.classes_).index(label)]
        for rank, j in enumerate(np.argsort(-coefs)[:k], start=1):
            rows.append({"intent": label, "rank": rank, "feature": vocab[j], "coefficient": float(coefs[j])})
    return pd.DataFrame(rows)


def guard_output_dataset(P: OutputPaths, fingerprint: dict) -> None:
    """Refuse to overwrite an output folder that holds results for a different dataset (e.g. running the
    V5 data with the V4 config would otherwise replace the published V4 results)."""
    existing = P.metrics / "experiment_config.json"
    if not existing.exists():
        return
    previous = load_json(existing).get("dataset", {}).get("content_sha256")
    if previous and previous != fingerprint["content_sha256"]:
        raise RuntimeError(f"{P.rel(P.results)} holds results for a different dataset "
                           f"(content sha256 {previous[:12]}… vs {fingerprint['content_sha256'][:12]}…). "
                           "Use a config with its own `outputs:` folders.")


def run_experiment(config_path=None) -> dict:
    t_start = time.perf_counter()
    cfg_path = resolve_path(config_path or DEFAULT_CONFIG)
    cfg = load_config(cfg_path)
    exp = cfg["experiment"]
    seed = exp["random_seed"]
    set_seed(seed)
    P = output_paths(cfg)
    plots.apply_style(cfg["plots"]["dpi"])
    labels, core = cfg["labels"], cfg["core_labels"]
    model_name, exp_id = exp["model_name"], exp["experiment_id"]
    diag = cfg["diagnostics"]
    token_pattern = cfg["preprocessing"]["token_pattern"]

    # ---------------------------------------------------------------- 1. data + audit
    print("[1/9] Loading and auditing dataset")
    df = load_dataset(cfg)
    validate_schema(df, cfg)
    fingerprint = dataset_fingerprint(df, cfg)
    guard_output_dataset(P, fingerprint)
    P.ensure()
    if cfg.get("validation"):
        validation = strict_dataset_checks(df, cfg)
        save_json(validation, P.metrics / "dataset_validation.json")
        pd.DataFrame(validation["checks"]).to_csv(P.metrics / "dataset_validation.csv", index=False)
        failed = [c["check"] for c in validation["checks"] if c["status"] == "fail"]
        if failed:
            raise DatasetError(f"Dataset validation failed before training: {failed} "
                               f"(see {P.rel(P.metrics / 'dataset_validation.csv')})")
    else:
        validation = None
    splits = get_splits(df, cfg)
    audit_summary, findings, details = audit_dataset(df, cfg)
    save_json(audit_summary | {"fingerprint": fingerprint}, P.metrics / "dataset_audit.json")
    findings.to_csv(P.metrics / "dataset_audit.csv", index=False)
    for name, frame in details.items():
        frame.to_csv(P.processed / f"{name}.csv", index=False)
    leak = leakage_table(df, cfg).set_index("row_id")
    keyed = add_audit_keys(df)
    keyed["leakage_flag"] = keyed["row_id"].map(leak["leakage_flag"]).fillna(False).astype(bool)
    keyed["max_jaccard_to_train"] = keyed["row_id"].map(leak["max_jaccard_to_train"])
    keyed.to_csv(P.processed / "dataset_with_audit_keys.csv", index=False)
    for name, frame in splits.as_dict().items():
        frame.to_csv(P.processed / f"split_{name}.csv", index=False)

    train, val, test, unk = splits.train, splits.validation, splits.test, splits.unknown
    assert UNKNOWN_INTENT not in set(train["intent"]), "UNKNOWN_INTENT leaked into training data"

    # ---------------------------------------------------------------- 2. tuning
    print("[2/9] Hyperparameter search on validation macro F1")
    t0 = time.perf_counter()
    grid_results, best_params = tune_hyperparameters(train, val, cfg, labels)
    grid_seconds = time.perf_counter() - t0
    grid_results.to_csv(P.metrics / "hyperparameter_search.csv", index=False)
    default_params = {"ngram_range": tuple(cfg["model"]["default_params"]["ngram_range"]),
                      "C": float(cfg["model"]["default_params"]["C"]),
                      "binary": bool(cfg["model"]["default_params"]["binary"])}

    # ---------------------------------------------------------------- 3. fit + evaluate variants
    print("[3/9] Fitting default and tuned models; evaluating all splits")
    variants: dict[str, dict] = {}
    split_frames = {"train": train, "validation": val, "test": test}
    clean_test = test[~test["row_id"].map(leak["leakage_flag"]).astype(bool)].reset_index(drop=True)
    for variant, params in (("default", default_params), ("tuned", best_params)):
        pipe, fit_seconds = fit_pipeline(build_pipeline(params, cfg), train["text"], train["intent"], labels)
        val_proba = reorder_proba(pipe, val["text"], labels)
        threshold = threshold_for_coverage(max_confidence(val_proba), cfg["rejection"]["target_known_coverage"])
        v = {"params": params, "pipeline": pipe, "fit_seconds": fit_seconds, "threshold": threshold,
             "metrics": {}, "proba": {}, "pred": {}}
        for split, frame in split_frames.items():
            proba = reorder_proba(pipe, frame["text"], labels)
            pred = np.asarray(labels, dtype=object)[proba.argmax(axis=1)]
            v["proba"][split], v["pred"][split] = proba, pred
            v["metrics"][split] = evaluate_predictions(frame["intent"], pred, proba, labels, split,
                                                       f"{model_name}_{variant}", exp_id, core_labels=core)
        proba = reorder_proba(pipe, clean_test["text"], labels)
        v["metrics"]["test_leakage_controlled"] = evaluate_predictions(
            clean_test["intent"], np.asarray(labels, dtype=object)[proba.argmax(axis=1)], proba, labels,
            "test_leakage_controlled", f"{model_name}_{variant}", exp_id, core_labels=core)
        v["proba"]["unknown"] = reorder_proba(pipe, unk["text"], labels)
        v["rejection_test"] = rejection_outcomes(test["intent"], v["proba"]["test"], v["proba"]["unknown"],
                                                 labels, threshold)
        variants[variant] = v

    # Test phrases repeat (up to ~10x); add a one-row-per-phrase view and phrase-clustered CIs.
    test_groups = test["text"].map(canonical_key).to_numpy()
    first_of_phrase = ~pd.Series(test_groups).duplicated().to_numpy()
    for variant, v in variants.items():
        v["metrics"]["test_unique_phrases"] = evaluate_predictions(
            test["intent"][first_of_phrase], v["pred"]["test"][first_of_phrase],
            v["proba"]["test"][first_of_phrase], labels, "test_unique_phrases", f"{model_name}_{variant}",
            exp_id, core_labels=core)
    bootstrap = cluster_bootstrap(test["intent"], {"tuned": variants["tuned"]["pred"]["test"],
                                                   "default": variants["default"]["pred"]["test"]},
                                  labels, test_groups, n_resamples=cfg["bootstrap"]["n_resamples"], seed=seed)
    save_json(bootstrap, P.metrics / "bootstrap_ci_test.json")

    tuned = variants["tuned"]
    pipe, threshold = tuned["pipeline"], tuned["threshold"]
    for split in ["train", "validation", "test", "test_leakage_controlled", "test_unique_phrases"]:
        save_json(tuned["metrics"][split], P.metrics / f"metrics_{split}.json")
        save_json(variants["default"]["metrics"][split], P.metrics / f"metrics_{split}_default_model.json")
    pd.concat([per_class_frame(v["metrics"][s]).assign(variant=name)
               for name, v in variants.items()
               for s in ["train", "validation", "test", "test_leakage_controlled", "test_unique_phrases"]]
              ).to_csv(P.metrics / "per_class_metrics.csv", index=False)
    cm, cm_norm = confusion_frames(tuned["metrics"]["test"])
    cm.to_csv(P.metrics / "confusion_matrix.csv")
    cm_norm.to_csv(P.metrics / "confusion_matrix_normalized.csv")
    pd.concat([pd.DataFrame(tuned["metrics"][s]["per_class"])[["intent", "support", "roc_auc_ovr", "average_precision"]]
               .assign(split=s) for s in ["train", "validation", "test"]]
              ).to_csv(P.metrics / "roc_auc_per_class.csv", index=False)

    # ---------------------------------------------------------------- 4. rejection
    print("[4/9] Unknown-intent rejection protocol")
    val_conf = max_confidence(tuned["proba"]["validation"])
    rej_val = rejection_outcomes(val["intent"], tuned["proba"]["validation"], np.empty((0, len(labels))),
                                 labels, threshold)
    rej_test = tuned["rejection_test"]
    s = cfg["rejection"]["sweep_thresholds"]
    thresholds = np.round(np.arange(s["start"], s["stop"] + 1e-9, s["step"]), 4)
    sweep = threshold_sweep(test["intent"], tuned["proba"]["test"], tuned["proba"]["unknown"], labels, thresholds)
    sweep.to_csv(P.metrics / "rejection_threshold_sweep_exploratory.csv", index=False)

    # Supplementary unknown-aware threshold: calibrate on half the unknown phrases, report on the other half.
    unk_groups = unk["text"].map(canonical_key)
    gss = GroupShuffleSplit(n_splits=1, train_size=cfg["rejection"]["unknown_calibration_fraction"], random_state=seed)
    cal_idx, hold_idx = next(gss.split(unk, groups=unk_groups))
    unk_proba = tuned["proba"]["unknown"]
    aware_thr = balanced_threshold(tuned["proba"]["validation"], unk_proba[cal_idx], thresholds)
    supplementary = {
        "description": ("Unknown-aware threshold chosen on validation knowns + a calibration half of the "
                        "unknown phrases (grouped by normalized phrase), reported on test knowns + the "
                        "disjoint holdout half. Supplementary only: the 300 unknowns were also used for the "
                        "primary frozen-threshold evaluation, and both halves share a synthetic generator."),
        "n_unknown_calibration": int(len(cal_idx)), "n_unknown_holdout": int(len(hold_idx)),
        "shared_phrases_between_halves": int(len(set(unk_groups.iloc[cal_idx]) & set(unk_groups.iloc[hold_idx]))),
        "objective": "maximize 0.5*known_val_coverage + 0.5*unknown_calibration_rejection",
        "selected_threshold": aware_thr,
        "holdout_results_unknown_aware_threshold": rejection_outcomes(
            test["intent"], tuned["proba"]["test"], unk_proba[hold_idx], labels, aware_thr),
        "holdout_results_frozen_primary_threshold": rejection_outcomes(
            test["intent"], tuned["proba"]["test"], unk_proba[hold_idx], labels, threshold),
    }
    rejection_metrics = {
        "rule": "if max_probability < threshold: UNKNOWN_INTENT else argmax intent",
        "threshold_selection": {
            "method": "known-intent validation coverage target (prespecified before viewing unknown labels)",
            "target_known_coverage": cfg["rejection"]["target_known_coverage"],
            "threshold": threshold,
            "validation_known_coverage_achieved": float((val_conf >= threshold).mean()),
        },
        "validation_known_only": rej_val,
        "primary_frozen_threshold_test_plus_unknown": rej_test,
        "unknown_unique_phrases": int(unk_groups.nunique()),
        "unknown_rejection_rate_unique_phrases": float(
            pd.Series(max_confidence(unk_proba) < threshold).groupby(unk_groups.to_numpy()).mean().mean()),
        "default_model_frozen_threshold": variants["default"]["rejection_test"],
        "exploratory_sweep_note": ("rejection_threshold_sweep_exploratory.csv and figure 10 use the "
                                   "evaluation-only unknowns at every threshold; they are exploratory, not an "
                                   "independent test, and were not used to choose the primary threshold."),
        "supplementary_unknown_aware": supplementary,
    }
    save_json(rejection_metrics, P.metrics / "rejection_metrics.json")

    # ---------------------------------------------------------------- 5. persistence + runtime
    print("[5/9] Saving models and measuring runtime")
    version = exp["model_version"]
    common_meta = {"model_name": model_name, "experiment_id": exp_id, "model_version": version,
                   "created_utc": utc_now(), "dataset": fingerprint, "packages": package_versions(),
                   "random_seed": seed}
    paths = {}
    for variant, v in variants.items():
        paths[variant] = save_bundle(
            v["pipeline"], P.models / f"{exp_id}_{variant}_v{version}.joblib", class_order=labels,
            threshold=v["threshold"],
            metadata=common_meta | {"variant": variant, "params": v["params"], "training_seconds": v["fit_seconds"],
                                    "validation_macro_f1": v["metrics"]["validation"]["macro_f1"]})
    lat_cfg = cfg["latency"]
    test_texts = list(test["text"])
    runtime = {"environment": environment_info(), "packages": package_versions(),
               "methodology": ("time.perf_counter around one end-to-end predict_proba call on raw text "
                               "(normalization + vectorization + LR). Warm-up calls discarded. Single-request "
                               "batches sample random test utterances; batch measurement uses all 270 test rows. "
                               "Single process, no other workload controls; wall-clock times vary by machine.")}
    for variant, v in variants.items():
        lat1, samples1 = measure_latency(v["pipeline"].predict_proba, test_texts, batch_size=1,
                                         warmup=lat_cfg["warmup"], repeats=lat_cfg["single_repeats"], seed=seed)
        latb, samplesb = measure_latency(v["pipeline"].predict_proba, test_texts, batch_size=len(test_texts),
                                         warmup=10, repeats=lat_cfg["batch_repeats"], seed=seed)
        clf = v["pipeline"].named_steps["clf"]
        runtime[variant] = {
            "params": v["params"],
            "training_seconds": v["fit_seconds"],
            "latency_single_request": lat1, "latency_batch": latb,
            "model_file": P.rel(paths[variant]),
            "model_size_mb": paths[variant].stat().st_size / 1024**2,
            "vocabulary_size": len(v["pipeline"].named_steps["bow"].vocabulary_),
            "n_parameters": int(clf.coef_.size + clf.intercept_.size),
            "coef_shape": list(clf.coef_.shape),
        }
        if variant == "tuned":
            latency_samples = {"Single request (batch=1)": samples1, f"Batch of {len(test_texts)}": samplesb}
    runtime["hyperparameter_search_seconds"] = grid_seconds
    save_json(runtime, P.metrics / "runtime_metrics.json")

    # ---------------------------------------------------------------- 6. predictions
    print("[6/9] Exporting row-level predictions")
    preds = {}
    for split, frame in split_frames.items():
        p = _prediction_frame(frame, tuned["proba"][split], labels, threshold)
        p["leakage_flag"] = p["row_id"].map(leak["leakage_flag"]).fillna(False).astype(bool)
        preds[split] = p
        p.to_csv(P.predictions / f"{split}_predictions.csv", index=False)
    pu = _prediction_frame(unk, unk_proba, labels, threshold)
    pu["correct"] = pu["rejected"]  # for unknowns, "correct" means correctly rejected
    pu["unknown_calibration_half"] = False
    pu.loc[cal_idx, "unknown_calibration_half"] = True
    preds["unknown"] = pu
    pu.to_csv(P.predictions / "unknown_predictions.csv", index=False)

    probe_df = pd.DataFrame(DIAGNOSTIC_PROBES, columns=["text", "expected", "probe_purpose"])
    probe_res = pd.DataFrame(predict_with(pipe, probe_df["text"], labels, threshold))
    probe_probs = pd.DataFrame(list(probe_res.pop("probabilities"))).add_prefix("prob_")
    probe_df = pd.concat([probe_df, probe_res.drop(columns="text"), probe_probs], axis=1)
    dataset_keys = set(keyed["canonical"])
    probe_df["phrase_in_dataset"] = probe_df["text"].map(canonical_key).isin(dataset_keys)
    probe_df["matches_expected"] = np.where(probe_df["expected"] == "AMBIGUOUS", None,
                                            probe_df["final_intent"] == probe_df["expected"])
    probe_df.insert(0, "probe_type", "qualitative_diagnostic_probe")
    probe_df.to_csv(P.predictions / "diagnostic_predictions.csv", index=False)

    word_order = []
    for a, b in WORD_ORDER_PROBES:
        ra, rb = predict_with(pipe, [a, b], labels, threshold)
        word_order.append({"text_a": a, "pred_a": ra["predicted_intent"], "conf_a": ra["max_confidence"],
                           "text_b": b, "pred_b": rb["predicted_intent"], "conf_b": rb["max_confidence"],
                           "same_bag_of_words": sorted(tokenize(normalize_text(a))) == sorted(tokenize(normalize_text(b))),
                           "max_abs_prob_diff": float(np.max(np.abs(np.array(list(ra["probabilities"].values()))
                                                                    - np.array(list(rb["probabilities"].values())))))})
    word_order = pd.DataFrame(word_order)
    word_order.to_csv(P.predictions / "word_order_probes.csv", index=False)

    # ---------------------------------------------------------------- 7. figures
    print("[7/9] Generating figures and learning curve")
    F = P.figures
    tm = tuned["metrics"]["test"]
    plots.confusion_matrix_plot(cm, F / "01_confusion_matrix_counts.png", "Test confusion matrix (counts)", False)
    plots.confusion_matrix_plot(cm_norm, F / "02_confusion_matrix_normalized.png",
                                "Test confusion matrix (row-normalized recall)", True)
    plots.per_class_metrics_plot(pd.DataFrame(tm["per_class"]), F / "03_per_class_metrics.png",
                                 "Per-class precision, recall and F1 (test)")
    comp_metrics = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "weighted_f1", "macro_roc_auc_ovr", "macro_average_precision"]
    comparison = pd.DataFrame({s: [tuned["metrics"][s][m] for m in comp_metrics]
                               for s in ["train", "validation", "test", "test_leakage_controlled"]},
                              index=comp_metrics).astype(float)
    comparison.columns = ["train", "validation", "test", "test (leakage-controlled)"]
    plots.split_comparison_plot(comparison, F / "04_split_comparison.png", "Train / validation / test comparison")
    comparison.to_csv(P.metrics / "split_comparison.csv")
    plots.roc_curves_plot(test["intent"], tuned["proba"]["test"], labels, F / "05_roc_curves.png",
                          "One-vs-rest ROC curves (test)").to_csv(P.metrics / "roc_curves_data.csv", index=False)
    plots.pr_curves_plot(test["intent"], tuned["proba"]["test"], labels, F / "06_precision_recall_curves.png",
                         "One-vs-rest precision-recall curves (test)").to_csv(P.metrics / "pr_curves_data.csv", index=False)

    groups = train["text"].map(canonical_key)
    cv = StratifiedGroupKFold(n_splits=cfg["learning_curve"]["cv_folds"], shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sizes, tr_scores, cv_scores = learning_curve(
            build_pipeline(best_params, cfg), list(train["text"]), list(train["intent"]), groups=groups,
            train_sizes=cfg["learning_curve"]["train_sizes"], cv=cv, scoring="f1_macro",
            shuffle=True, random_state=seed, error_score=np.nan, n_jobs=1)
    lc = pd.DataFrame({"train_size": sizes,
                       "train_mean": np.nanmean(tr_scores, 1), "train_std": np.nanstd(tr_scores, 1),
                       "cv_mean": np.nanmean(cv_scores, 1), "cv_std": np.nanstd(cv_scores, 1),
                       "folds_failed": np.isnan(cv_scores).sum(1)})
    plots.learning_curve_plot(lc, F / "07_learning_curve.png",
                              "Learning curve (training split only, 5-fold CV grouped by phrase)")
    lc.to_csv(P.metrics / "learning_curve.csv", index=False)

    test_conf = max_confidence(tuned["proba"]["test"])
    test_correct = preds["test"]["correct"].to_numpy()
    plots.confidence_distribution_plot(test_conf, test_correct, threshold, F / "08_confidence_distribution.png",
                                       "Confidence: correct vs incorrect (test)"
                                       ).to_csv(P.metrics / "confidence_distribution_test.csv", index=False)
    ece, cal_bins = expected_calibration_error(test["intent"], tuned["pred"]["test"], test_conf)
    plots.calibration_plot(cal_bins, ece, F / "09_calibration_curve.png", "Reliability diagram (test, top label)")
    cal_bins.to_csv(P.metrics / "calibration_bins_test.csv", index=False)
    plots.rejection_threshold_plot(sweep, threshold, F / "10_rejection_threshold.png",
                                   "Rejection threshold sensitivity")
    lat_df = plots.latency_plot(latency_samples, F / "11_inference_latency.png",
                                "Inference latency (tuned model, CPU, end-to-end)")
    lat_df.to_csv(P.metrics / "latency_samples.csv", index=False)
    dist = pd.crosstab(df["intent"], df["split"]).reindex(index=labels + [UNKNOWN_INTENT],
                                                         columns=["train", "validation", "test", "evaluation_only"],
                                                         fill_value=0)
    plots.class_distribution_plot(dist, F / "12_class_distribution.png", "Class distribution across splits")
    dist.to_csv(P.metrics / "class_distribution.csv")
    top = _top_features(pipe, labels, diag["top_features"])
    plots.top_features_plot(top, labels, F / "13_top_features.png",
                            f"Top {diag['top_features']} positive LR coefficients per intent (associations, not causes)")
    top.to_csv(P.metrics / "top_features.csv", index=False)

    # ---------------------------------------------------------------- 8. supplementary core-8 model
    print("[8/9] Supplementary core-8 model and error analysis")
    tr8, va8, te8 = (f[f["intent"].isin(core)].reset_index(drop=True) for f in (train, val, test))
    pipe8, secs8 = fit_pipeline(build_pipeline(best_params, cfg), tr8["text"], tr8["intent"], core)
    core8 = {}
    for split, frame in (("train", tr8), ("validation", va8), ("test", te8)):
        pr = reorder_proba(pipe8, frame["text"], core)
        core8[split] = evaluate_predictions(frame["intent"], np.asarray(core, dtype=object)[pr.argmax(1)], pr,
                                            core, split, f"{model_name}_core8", exp_id)
    thr8 = threshold_for_coverage(max_confidence(reorder_proba(pipe8, va8["text"], core)),
                                  cfg["rejection"]["target_known_coverage"])
    rej8 = rejection_outcomes(te8["intent"], reorder_proba(pipe8, te8["text"], core),
                              reorder_proba(pipe8, unk["text"], core), core, thr8)
    save_json({"note": "Supplementary 8-class model trained on core intents only with the tuned "
                       "hyperparameters (not re-tuned). Primary results are the 9-class model.",
               "params": best_params, "training_seconds": secs8, "metrics": core8, "rejection": rej8},
              P.metrics / "metrics_core8_model.json")

    # ---------------------------------------------------------------- 9. error analysis + summary + reports
    vocab = set(pipe.named_steps["bow"].vocabulary_)
    for split in ["train", "validation", "test"]:
        oov, frac = _oov_info(preds[split]["text"], vocab, token_pattern)
        preds[split]["oov_tokens"] = [" ".join(o) for o in oov]
        preds[split]["oov_fraction"] = frac
        preds[split]["n_tokens"] = preds[split]["text"].map(lambda t: len(tokenize(normalize_text(t), token_pattern)))
    oov_u, frac_u = _oov_info(pu["text"], vocab, token_pattern)
    pu["oov_tokens"], pu["oov_fraction"] = [" ".join(o) for o in oov_u], frac_u

    rt = runtime["tuned"]
    config_sha = sha256_file(cfg_path)
    summary_rows = []
    for variant, v in variants.items():
        m, r = v["metrics"], v["rejection_test"]
        summary_rows.append(summary_row(
            model_name=f"{model_name}_{variant}", experiment_id=exp_id, variant=variant,
            dataset_version=cfg["data"]["dataset_version"], dataset_sha256=fingerprint["file_sha256"],
            config_sha256=config_sha, random_seed=seed,
            train_accuracy=m["train"]["accuracy"], validation_accuracy=m["validation"]["accuracy"],
            test_accuracy=m["test"]["accuracy"], train_macro_f1=m["train"]["macro_f1"],
            validation_macro_f1=m["validation"]["macro_f1"], test_macro_f1=m["test"]["macro_f1"],
            test_weighted_f1=m["test"]["weighted_f1"], test_macro_roc_auc=m["test"]["macro_roc_auc_ovr"],
            test_log_loss=m["test"]["log_loss"], test_balanced_accuracy=m["test"]["balanced_accuracy"],
            test_core8_macro_f1=m["test"]["core8_breakdown"]["macro_f1"],
            test_leakage_controlled_macro_f1=m["test_leakage_controlled"]["macro_f1"],
            rejection_threshold=v["threshold"], known_intent_coverage=r["known_intent_coverage"],
            unknown_rejection_rate=r["unknown_rejection_rate"], unknown_false_accept_rate=r["unknown_false_accept_rate"],
            inference_p50_ms=runtime[variant]["latency_single_request"]["median_ms"],
            inference_p95_ms=runtime[variant]["latency_single_request"]["p95_ms"],
            inference_p99_ms=runtime[variant]["latency_single_request"]["p99_ms"],
            throughput_per_s=runtime[variant]["latency_batch"]["throughput_predictions_per_s"],
            training_seconds=v["fit_seconds"], model_size_mb=runtime[variant]["model_size_mb"],
            n_parameters=runtime[variant]["n_parameters"], vocabulary_size=runtime[variant]["vocabulary_size"],
            notes=f"{params_id(v['params'])}; 9 classes; threshold from 95% validation coverage"
                  + ("; PRIMARY" if variant == "tuned" else "; untuned reference")))
    clf8 = pipe8.named_steps["clf"]
    summary_rows.append(summary_row(
        model_name=f"{model_name}_core8", experiment_id=exp_id, variant="core8_supplementary",
        dataset_version=cfg["data"]["dataset_version"], dataset_sha256=fingerprint["file_sha256"],
        config_sha256=config_sha, random_seed=seed,
        train_accuracy=core8["train"]["accuracy"], validation_accuracy=core8["validation"]["accuracy"],
        test_accuracy=core8["test"]["accuracy"], train_macro_f1=core8["train"]["macro_f1"],
        validation_macro_f1=core8["validation"]["macro_f1"], test_macro_f1=core8["test"]["macro_f1"],
        test_weighted_f1=core8["test"]["weighted_f1"], test_macro_roc_auc=core8["test"]["macro_roc_auc_ovr"],
        test_log_loss=core8["test"]["log_loss"], test_balanced_accuracy=core8["test"]["balanced_accuracy"],
        test_core8_macro_f1=core8["test"]["macro_f1"], rejection_threshold=thr8,
        known_intent_coverage=rej8["known_intent_coverage"], unknown_rejection_rate=rej8["unknown_rejection_rate"],
        unknown_false_accept_rate=rej8["unknown_false_accept_rate"], training_seconds=secs8,
        n_parameters=int(clf8.coef_.size + clf8.intercept_.size),
        vocabulary_size=len(pipe8.named_steps["bow"].vocabulary_),
        notes=f"{params_id(best_params)}; 8 core classes only; not saved; latency not measured"))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(P.reports / "experiment_summary.csv", index=False)

    experiment_config = {
        "experiment": exp, "created_utc": utc_now(), "config_path": str(cfg_path), "config_sha256": config_sha,
        "config": cfg, "dataset": fingerprint, "packages": package_versions(), "environment": environment_info(),
        "default_params": default_params, "selected_params": best_params,
        "selection": "highest validation macro F1; ties broken by lower validation log loss; test never used",
        "rejection_threshold": threshold, "model_artifacts": {k: str(p) for k, p in paths.items()},
        "splits_used": {"fit": "train (supported intents only)", "model_selection": "validation",
                        "threshold_selection": "validation known intents only",
                        "final_evaluation": "test (+ evaluation_only unknowns for rejection)"},
    }
    save_json(experiment_config, P.metrics / "experiment_config.json")

    ctx = {
        "cfg": cfg, "labels": labels, "core": core, "fingerprint": fingerprint, "audit": audit_summary,
        "findings": findings, "audit_details": details, "grid": grid_results, "best_params": best_params,
        "default_params": default_params, "variants": variants, "threshold": threshold,
        "rejection": rejection_metrics, "runtime": runtime, "preds": preds, "probes": probe_df,
        "word_order": word_order, "comparison": comparison, "learning_curve": lc, "ece": ece,
        "core8": core8, "core8_rejection": rej8, "top_features": top, "summary": summary, "bootstrap": bootstrap,
        "sweep": sweep, "diag": diag, "money_moving": MONEY_MOVING, "paths": paths,
        "out": P, "validation": validation, "test_correct": int(preds["test"]["correct"].sum()),
    }
    if cfg.get("threshold_sweep"):
        from .threshold_sweep import run_threshold_sweep
        ctx["threshold_sweep"] = run_threshold_sweep(
            cfg, P, labels, default_threshold=threshold, test=preds["test"], unknown=preds["unknown"],
            test_proba=tuned["proba"]["test"], unknown_proba=unk_proba,
            validation_proba=tuned["proba"]["validation"], financial=MONEY_MOVING)
    print("[9/9] Writing reports")
    write_reports(ctx)
    if cfg.get("comparison"):
        from .compare import write_version_comparison
        write_version_comparison(cfg, P, ctx)
    elapsed = time.perf_counter() - t_start
    print(f"Done in {elapsed:.1f}s")
    return ctx


def predict_with(pipe, texts, class_order: list[str], threshold: float) -> list[dict]:
    texts = list(texts)
    proba = reorder_proba(pipe, texts, class_order)
    final = apply_rejection(proba, class_order, threshold)
    labels = np.asarray(class_order, dtype=object)
    return [{"text": t, "predicted_intent": labels[proba[i].argmax()], "final_intent": final[i],
             "max_confidence": float(proba[i].max()),
             "probabilities": dict(zip(class_order, proba[i].round(6).tolist()))}
            for i, t in enumerate(texts)]
