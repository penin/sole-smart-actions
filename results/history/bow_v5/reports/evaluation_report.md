# Evaluation Report — Sole Smart Actions BoW + Logistic Regression Baseline

_Generated 2026-11-12T02:55:57+00:00 · experiment `bow_logreg_v5` · model version 1.0.0 · seed 42 · dataset `v5_deduplicated` (SHA-256 `8dc9c89cbb8a…`)._

All numbers below were produced by actually training and evaluating the models in this run. Result types are labelled: **in-sample** (train), **selection** (validation), **held-out** (test), **exploratory** (re-uses evaluation-only unknowns) and **diagnostic** (hand-written probes).

## 1. Executive summary

- **Raw Out-of-Sample Test Accuracy (No Confidence Threshold): 0.937** (253 of 270 legitimate test commands classified correctly; no prediction excluded).
- **Selected model** (best validation macro F1): `ngram(1,2) C=10 binary`. Held-out test accuracy **0.937**, macro F1 **0.937** (95% phrase-clustered CI 0.905–0.964), macro ROC-AUC **0.995**.
- Train macro F1 1.000 vs validation 0.917 and test 0.937: a train–test accuracy gap of 6.3 points. 11.5% of test rows contain a word never seen in training; they account for 88.2% of test errors. BoW cannot map an unseen word to a known meaning.
- The **untuned default** (`ngram(1,1) C=1 counts`) scored lower on validation (0.845) and about the same on test (0.936 vs 0.937). The paired bootstrap difference (tuned − default) is +0.000 [-0.014, +0.016]. With 30 validation rows per class, model selection is noisy. The tuned model stays the primary result because the test set was not used for selection.
- **Unknown-intent rejection is weak.** With the threshold frozen at 0.342 (95% known-intent validation coverage), 29.3% of the 300 unsupported requests are rejected and 70.7% are falsely accepted. Known-intent test coverage is 96.3%. The maximum-probability rule is not a safe out-of-scope guard here.
- TRANSFER_MONEY ↔ BILL_PAY confusions on validation + test: 3 rows; no training TRANSFER_MONEY rows are predicted BILL_PAY. 23 unsupported requests were accepted as a money-moving intent (TRANSFER_MONEY, BILL_PAY or WITHDRAW_MONEY).
- The model is tiny and fast: 2584 vocabulary terms, 23,265 parameters, 222.1 KB on disk, 175 ms to train, single-request median latency 0.280 ms (P95 0.444 ms) on CPU.

## 2. Dataset

3,000 synthetic utterances (sheet `Training Data`): 9 supported intents × 300 (240 train / 30 validation / 30 test each) and 300 `UNKNOWN_INTENT` rows reserved for rejection evaluation (`evaluation_only`). Eight intents are `core`; `CHECK_SPENDING` is the `stretch` intent. The supplied splits were used unchanged. Figure: `results_v5/figures/12_class_distribution.png`.

## 3. Data-quality findings

- No missing values, empty texts, invalid labels or contradictory labels.
- Repetition: 7 redundant rows under a punctuation-insensitive key; 270 distinct phrases among 270 test rows.
- Train overlap: 0 test rows repeat a training phrase exactly or after normalization, 39 once numbers are masked (validation: 1 / 23). Reported, not removed; a leakage-controlled subset (231 test rows) is evaluated.
- Moderate vocabulary shift between train and held-out partitions: 11.5% of test rows contain an unseen word (see §13).
Full details: `results_v5/reports/data_quality_report.md`, `results_v5/metrics/dataset_audit.{json,csv}`.

## 4. Model architecture and preprocessing

`raw text → TextNormalizer → CountVectorizer → LogisticRegression → 9 class probabilities → threshold rejection`

- **Normalization:** Unicode NFKC, lowercase, whitespace collapse. No stemming, lemmatization or stop-word removal.
- **Tokenization:** `(?u)\$|\b\w+\b` keeps `$`, digits and one-character tokens (sklearn's default pattern would drop `$`, `5`, `2`). Amounts stay as literal tokens; masking numbers is a candidate improvement, not applied here.
- **Classifier:** multinomial logistic regression, `lbfgs` solver, L2 penalty, max_iter 5000 (convergence warnings are raised as errors).
- Vectorizer and classifier are a single sklearn `Pipeline`, so the vocabulary is fitted only on the training split (verified by unit tests).

## 5. Training methodology

- Fit on `split = train` rows with a supported intent (2,160 rows). `UNKNOWN_INTENT` is never a class.
- Hyperparameters chosen on `validation` macro F1 (ties broken by validation log loss).
- The rejection threshold is chosen on known-intent validation rows only.
- `test` and the unknown set are scored once, after everything is frozen.
- Canonical class order (shared with all team models): `TRANSFER_MONEY`, `CHECK_BALANCE`, `VIEW_TRANSACTIONS`, `WITHDRAW_MONEY`, `BILL_PAY`, `HELP_SUPPORT`, `MANAGE_CONTACTS`, `VIEW_GROUP`, `CHECK_SPENDING`.

## 6. Hyperparameter tuning

Grid: n-gram ∈ {(1,1), (1,2)} × C ∈ {0.1, 1, 10} × {counts, binary} = 12 configurations (`results_v5/metrics/hyperparameter_search.csv`).

| rank | params_id | train_macro_f1 | validation_macro_f1 | validation_accuracy | validation_log_loss | vocabulary_size |
|---|---|---|---|---|---|---|
| 1 | ngram12_C10_binary | 1.000 | 0.917 | 0.919 | 0.370 | 2,584 |
| 2 | ngram12_C10_count | 1.000 | 0.917 | 0.919 | 0.371 | 2,584 |
| 3 | ngram12_C1_count | 1.000 | 0.854 | 0.859 | 0.439 | 2,584 |
| 4 | ngram12_C1_binary | 1.000 | 0.854 | 0.859 | 0.444 | 2,584 |
| 5 | ngram11_C0.1_count | 0.987 | 0.854 | 0.859 | 0.695 | 374 |
| 6 | ngram11_C1_count | 1.000 | 0.845 | 0.852 | 0.409 | 374 |
| 7 | ngram11_C1_binary | 1.000 | 0.845 | 0.852 | 0.415 | 374 |
| 8 | ngram12_C0.1_count | 0.991 | 0.840 | 0.844 | 0.695 | 2,584 |
| 9 | ngram11_C0.1_binary | 0.987 | 0.838 | 0.841 | 0.700 | 374 |
| 10 | ngram12_C0.1_binary | 0.991 | 0.836 | 0.841 | 0.698 | 2,584 |
| 11 | ngram11_C10_count | 1.000 | 0.814 | 0.822 | 0.352 | 374 |
| 12 | ngram11_C10_binary | 1.000 | 0.814 | 0.822 | 0.361 | 374 |

Selected: `ngram(1,2) C=10 binary`. Validation macro F1 spans only 0.814–0.917 across the grid. Best validation macro F1 by C: C=0.1: 0.854, C=1: 0.854, C=10: 0.917. With 270 validation rows, one command is worth 0.37 accuracy points, so small gaps between configurations are noise.

**Default vs tuned** (default = unigram counts, C = 1):

| model | train_F1 | val_F1 | test_acc | test_F1 | test_F1_leak_ctrl | test_ROC_AUC | test_log_loss |
|---|---|---|---|---|---|---|---|
| default (ngram(1,1) C=1 counts) | 1.000 | 0.845 | 0.937 | 0.936 | 0.924 | 0.994 | 0.325 |
| tuned (ngram(1,2) C=10 binary) | 1.000 | 0.917 | 0.937 | 0.937 | 0.926 | 0.995 | 0.247 |

Phrase-clustered bootstrap on test (cluster bootstrap over 270 groups, 2000 resamples, 95% percentile intervals): tuned macro F1 0.937 [0.905, 0.964], default 0.936 [0.904, 0.964]; difference +0.000 [-0.014, +0.016], positive in 52.2% of resamples. 

## 7. In-sample vs out-of-sample performance (selected model)

| split | n | accuracy | balanced_acc | macro_P | macro_R | macro_F1 | weighted_F1 | log_loss | macro_ROC_AUC | weighted_ROC_AUC | macro_AP | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train (in-sample) | 2,160 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.007 | 1.000 | 1.000 | 1.000 | 0.007 |
| validation (used for selection) | 270 | 0.919 | 0.919 | 0.936 | 0.919 | 0.917 | 0.917 | 0.370 | 0.990 | 0.990 | 0.945 | 0.103 |
| test (held-out) | 270 | 0.937 | 0.937 | 0.943 | 0.937 | 0.937 | 0.937 | 0.247 | 0.995 | 0.995 | 0.979 | 0.065 |
| test, leakage-controlled | 231 | 0.926 | 0.929 | 0.930 | 0.929 | 0.926 | 0.926 | 0.288 | 0.993 | 0.994 | 0.970 | 0.075 |
| test, one row per phrase | 270 | 0.937 | 0.937 | 0.943 | 0.937 | 0.937 | 0.937 | 0.247 | 0.995 | 0.995 | 0.979 | 0.065 |

Figure: `04_split_comparison.png`. The learning curve (`07_learning_curve.png`, training split only, 5-fold CV grouped by phrase) reaches 0.994 CV macro F1 at 1728 examples, versus 0.917 on validation and 0.937 on test. Phrases resembling training phrases are easy; the gap comes from new phrasings, which more rows of the same templates are unlikely to fix.

## 8. Per-class performance (test, selected model)

| intent | scope | precision | recall | f1 | support | roc_auc_ovr | average_precision |
|---|---|---|---|---|---|---|---|
| TRANSFER_MONEY | core | 1.000 | 0.733 | 0.846 | 30 | 0.989 | 0.936 |
| CHECK_BALANCE | core | 0.968 | 1.000 | 0.984 | 30 | 1.000 | 1.000 |
| VIEW_TRANSACTIONS | core | 0.871 | 0.900 | 0.885 | 30 | 0.980 | 0.933 |
| WITHDRAW_MONEY | core | 0.968 | 1.000 | 0.984 | 30 | 1.000 | 1.000 |
| BILL_PAY | core | 0.968 | 1.000 | 0.984 | 30 | 1.000 | 1.000 |
| HELP_SUPPORT | core | 0.778 | 0.933 | 0.848 | 30 | 0.998 | 0.991 |
| MANAGE_CONTACTS | core | 0.933 | 0.933 | 0.933 | 30 | 0.993 | 0.968 |
| VIEW_GROUP | core | 1.000 | 1.000 | 1.000 | 30 | 1.000 | 1.000 |
| CHECK_SPENDING | stretch | 1.000 | 0.933 | 0.966 | 30 | 0.997 | 0.981 |

**Core-eight breakdown** (9-class model, rows whose true label is core; predictions of CHECK_SPENDING count as errors):

| split | n_samples | accuracy | macro_precision | macro_recall | macro_f1 | predicted_outside_core |
|---|---|---|---|---|---|---|
| train | 1,920 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| validation | 240 | 0.908 | 0.932 | 0.908 | 0.908 | 1 |
| test | 240 | 0.938 | 0.943 | 0.938 | 0.937 | 0 |

Supplementary 8-class model trained on core intents only (same hyperparameters, not re-tuned): test accuracy 0.938, macro F1 0.937, ROC-AUC 0.996 (`metrics_core8_model.json`).

## 9. Confusion-matrix analysis

Figures `01_confusion_matrix_counts.png` and `02_confusion_matrix_normalized.png`. Most frequent test confusions:

| actual_intent | predicted_intent | count | unique_phrases | example |
|---|---|---|---|---|
| TRANSFER_MONEY | HELP_SUPPORT | 7 | 7 | I owe joeli 100 |
| CHECK_SPENDING | VIEW_TRANSACTIONS | 2 | 2 | Show expenditure summary |
| MANAGE_CONTACTS | VIEW_TRANSACTIONS | 2 | 2 | Show beneficiaries |
| HELP_SUPPORT | BILL_PAY | 1 | 1 | I have a complaint |
| HELP_SUPPORT | MANAGE_CONTACTS | 1 | 1 | Need customer service |
| TRANSFER_MONEY | WITHDRAW_MONEY | 1 | 1 | I owe mary $50 |

Confusions cluster on a few phrasings. 88.2% of test errors contain a word absent from training (most frequent unseen held-out words: `look`, `owe`, `using`, `statement`, `talk`, `recent`). Unseen words are dropped, so the model falls back to whatever weak cues remain.

## 10. ROC-AUC and precision-recall analysis

Test macro one-vs-rest ROC-AUC 0.995 (weighted 0.995); macro average precision 0.979. ROC-AUC is high relative to accuracy because the correct intent is usually ranked near the top even when it is not top-1: top-2 accuracy is 0.959 and top-3 0.981, vs top-1 0.937. A clarification prompt offering the top two intents is a cheap mitigation. Per-class curves: `05_roc_curves.png`, `06_precision_recall_curves.png`; values in `roc_auc_per_class.csv`. Undefined metrics in this run: none.

**Calibration:** test ECE 0.065 (`09_calibration_curve.png`). Mean top-label confidence is 0.876 against accuracy 0.937: the model is **under-confident**. Threshold values do not transfer between models without recalibration.

## 11. Unknown-intent rejection

**Protocol.** Rule: `max_probability < threshold → UNKNOWN_INTENT`. The coverage target (95% of known validation intents accepted) was fixed before any unknown label was examined, giving threshold **0.3415** (validation coverage achieved 95.2%). The frozen threshold was applied once to test + the 300 evaluation-only unknowns.

| outcome | count |
|---|---|
| correct supported classification | 251 |
| incorrect supported classification | 9 |
| false rejection of supported | 10 |
| correct rejection of unsupported | 88 |
| false acceptance of unsupported | 212 |

- Known-intent coverage (test): **96.3%**; false rejection rate 3.7%; accuracy on accepted known requests 96.5%; end-to-end known accuracy 93.0%.
- Unknown rejection rate: **29.3%**; false acceptance rate **70.7%**. Per unique unknown phrase: 29.4% (299 phrases).
- Unknowns accepted, by predicted intent: HELP_SUPPORT 80, VIEW_TRANSACTIONS 48, MANAGE_CONTACTS 25, VIEW_GROUP 24, BILL_PAY 12, TRANSFER_MONEY 11, CHECK_BALANCE 9, CHECK_SPENDING 3.

The threshold-sensitivity curve (`10_rejection_threshold.png`, `rejection_threshold_sweep_exploratory.csv`) is **exploratory**: it re-uses the same evaluation-only unknowns at every threshold and is not an independent test.

**Supplementary unknown-aware threshold.** The 300 unknowns were split by normalized phrase into a calibration half (149 rows) and a disjoint holdout (151 rows, 0 shared phrases). Maximizing ½·coverage + ½·rejection on validation knowns + calibration unknowns selects threshold 0.77. On test knowns + holdout unknowns: unknown rejection 97.4% but known coverage only 78.1%. The primary threshold on the same holdout gives 29.1% / 96.3%. Confidence alone cannot separate known from unknown requests at an acceptable operating point. Caveat: both halves come from the same synthetic generator, and the holdout was also part of the primary evaluation, so this is not a fully untouched test.

## 11b. Threshold sweep (separate from raw accuracy)

Known commands are the 270 legitimate test rows; unknown commands are the 300 evaluation-only rows. *Correct routing* counts rejected legitimate commands as incorrect. The validation-calibrated default (34.15%) is the only methodological choice. **Every other row is exploratory** and must not be used to pick an "optimal" threshold, because the same test and unknown sets are scored at each one. The *No threshold* row equals raw test accuracy by construction (verified in code).

| threshold | known coverage | accepted (of 270) | correct & accepted | accuracy of accepted | correct routing (of 270) | legit rejected | unknown rejected | unknown → HELP_SUPPORT | unknown safe fallback | unknown → financial | → TRANSFER / WITHDRAW / BILL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No threshold | 100.0% | 270 | 253 | 93.7% | 93.7% | 0 | 0.0% | 96 | 32.0% | 45 | 22 / 7 / 16 |
| 25% | 98.1% | 265 | 253 | 95.5% | 93.7% | 5 | 10.3% | 90 | 40.3% | 36 | 22 / 0 / 14 |
| 30% | 96.3% | 260 | 251 | 96.5% | 93.0% | 10 | 22.3% | 86 | 51.0% | 26 | 14 / 0 / 12 |
| Default 34.15% (validation-calibrated) | 96.3% | 260 | 251 | 96.5% | 93.0% | 10 | 29.3% | 80 | 56.0% | 23 | 11 / 0 / 12 |
| 35% | 96.3% | 260 | 251 | 96.5% | 93.0% | 10 | 30.3% | 80 | 57.0% | 23 | 11 / 0 / 12 |
| 40% | 93.7% | 253 | 248 | 98.0% | 91.9% | 17 | 49.3% | 69 | 72.3% | 21 | 9 / 0 / 12 |
| 50% | 91.1% | 246 | 243 | 98.8% | 90.0% | 24 | 74.0% | 42 | 88.0% | 17 | 9 / 0 / 8 |
| 60% | 87.0% | 235 | 232 | 98.7% | 85.9% | 35 | 88.0% | 20 | 94.7% | 7 | 0 / 0 / 7 |

Files: `results_v5/threshold_analysis/threshold_sweep.csv`, `.json`, `dense_threshold_sweep.csv`, `decisions_at_default_threshold.csv`, `legitimate_rejection_by_intent.csv`; charts `threshold_comparison.png`, `coverage_vs_rejection.png`, `safe_fallback_comparison.png`, `financial_routing_risk.png`, `legitimate_rejection_by_intent.png`.

## 12. Computational performance

| Measure | Value |
|---|---|
| Training time (final fit, 2,160 rows) | 174.5 ms |
| Full grid search (12 fits + scoring) | 1.71 s |
| Single-request latency median / P95 / P99 | 0.280 / 0.444 / 0.661 ms |
| Single-request throughput (1 / median) | 3,574 req/s |
| Batch of 270 latency median / P95 | 2.86 / 3.56 ms |
| Batch throughput | 94,483 predictions/s |
| Serialized model (joblib bundle) | 222.1 KB |
| Vocabulary size | 2584 |
| Parameters (coefficients + intercepts) | 23,265 (9×2584 + 9) |
| Hardware | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 32 logical CPUs, 31.73 GB RAM, CPU only |
| Runtime | Windows-11-10.0.26200-SP0, Python 3.13.15, scikit-learn 1.9.1 |

Methodology: time.perf_counter around one end-to-end predict_proba call on raw text (normalization + vectorization + LR). Warm-up calls discarded. Single-request batches sample random test utterances; batch measurement uses all 270 test rows. Single process, no other workload controls; wall-clock times vary by machine. Single-request: 50 warm-up + 1000 measured calls; batch: 10 warm-up + 200 measured calls. Figure `11_inference_latency.png`.

## 13. Error analysis (summary)

| split | n | short_n | short_accuracy | longer_accuracy | rows_with_oov | share_rows_with_oov | oov_accuracy | in_vocab_accuracy | mean_oov_token_share |
|---|---|---|---|---|---|---|---|---|---|
| train | 2,160 | 71 | 1.000 | 1.000 | 0 | 0.000 | n/a | 1.000 | 0.000 |
| validation | 270 | 7 | 0.857 | 0.920 | 29 | 0.107 | 0.897 | 0.921 | 0.032 |
| test | 270 | 13 | 0.923 | 0.938 | 31 | 0.115 | 0.516 | 0.992 | 0.032 |

- 30 distinct validation/test phrases are misclassified **and** accepted by the rejection threshold. 6 held-out errors have confidence ≥ 0.9. Confident errors cannot be caught by any reasonable threshold.
- Word order: BoW gives identical probabilities to reordered commands (`transfer from mere to jone` vs `transfer from jone to mere`). Direction of payment is invisible to this model.
- Hand-written diagnostic probes are qualitative only (see `error_analysis.md`).

Full analysis: `results_v5/reports/error_analysis.md`.

## 14. Limitations and potential leakage

- **Synthetic data.** Utterances are generated from templates with filler suffixes. Real customer language in Fiji will include Fijian and Fiji Hindi words, code-switching, local slang and abbreviations (e.g. M-PAiSA, "vinaka", "bhaiya"), misspellings, SMS shorthand and speech-to-text errors. None of these is represented systematically here. Reported metrics should not be read as production accuracy.
- **Leakage.** 39 test rows match a training phrase (0 exactly or after normalization, 39 once numbers are masked). Leakage-controlled test macro F1 is 0.926 vs 0.937 on the full test set. Template-generated commands may also share structure across splits beyond what number masking detects.
- **Small held-out sets.** 270 test rows (270 distinct phrases); confidence intervals are wide (see §6), so small differences in macro F1 between models are not reliable.
- **Unknown evaluation.** The 300 unknowns are synthetic and sometimes padded with wallet words ("… on my account", "… on the app"). Rejection results are specific to this set.
- **BoW limits.** No word order, negation, synonyms or sub-word information. Unseen words are dropped silently.
- **Calibration.** Probabilities are under-confident; threshold values do not transfer across models without recalibration.

## 15. Recommendations for the next model

1. **Sub-word / character features** (char n-gram TF-IDF, fastText or a transformer tokenizer) to handle unseen words, misspellings and SMS forms. Unseen words appear in 88.2% of test errors, the dominant failure mode here.
2. **Pretrained sentence encoders or transformers** to map unseen paraphrases (`owe`, `beneficiaries`, `statement`) to known intents through semantic similarity.
3. **A dedicated out-of-scope detector** (e.g. trained on a separate pool of unsupported requests, energy/Mahalanobis scores or an embedding-distance check). Keep a genuinely untouched unknown test set.
4. **Calibrate probabilities** (temperature scaling on validation) before setting thresholds, and select thresholds on a separate calibration split.
5. **Safety confirmation for money-moving intents.** Require explicit confirmation of amount and payee for TRANSFER_MONEY, BILL_PAY and WITHDRAW_MONEY regardless of model confidence.
6. **Data:** collect or write more diverse paraphrases per intent, de-duplicate phrases, and use phrase-grouped splits so validation reflects test difficulty.
7. **Reuse this evaluation framework** (`sole_intent.evaluate`, `sole_intent.rejection`) and append to `experiment_summary.csv` so all models are compared on identical splits and metrics.
