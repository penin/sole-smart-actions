# Evaluation Report — Sole Smart Actions BoW + Logistic Regression Baseline

_Generated 2026-11-09T04:20:40+00:00 · experiment `bow_logreg_v1` · model version 1.0.0 · seed 42 · dataset `v4_cleaned` (SHA-256 `708e8f79ca47…`)._

All numbers below were produced by actually training and evaluating the models in this run. Result types are labelled: **in-sample** (train), **selection** (validation), **held-out** (test), **exploratory** (re-uses evaluation-only unknowns) and **diagnostic** (hand-written probes).

## 1. Executive summary

- **Selected model** (best validation macro F1): unigram counts, C = 0.1 (`ngram(1,1) C=0.1 counts`). Held-out test accuracy **0.793**, macro F1 **0.796** (95% phrase-clustered CI 0.712–0.864), macro ROC-AUC **0.954**.
- Train macro F1 0.987 vs validation 0.735: a large generalization gap. Validation/test contain paraphrase templates whose key words never occur in training (46.3% of test rows contain an unseen word). BoW cannot map an unseen word to a known meaning.
- The **untuned default** (`ngram(1,1) C=1 counts`) scored lower on validation (0.673) but higher on test (0.855). The paired bootstrap difference (tuned − default) is -0.058 [-0.109, -0.012]. With 30 validation rows per class, many of them repeats, model selection is noisy. The tuned model stays the primary result because the test set was not used for selection.
- **Unknown-intent rejection is weak.** With the threshold frozen at 0.196 (95% known-intent validation coverage), 14.7% of the 300 unsupported requests are rejected and 85.3% are falsely accepted. Known-intent test coverage is 90.7%. The maximum-probability rule is not a safe out-of-scope guard here.
- TRANSFER_MONEY ↔ BILL_PAY confusions on validation + test: 0 rows, but 15 *training* TRANSFER_MONEY rows (e.g. `Pay joeli 80`) are predicted BILL_PAY even in-sample: `pay <person> <amount>` collides with the strongest BILL_PAY cue, `pay`. 29 unsupported requests were accepted as a money-moving intent (TRANSFER_MONEY, BILL_PAY or WITHDRAW_MONEY).
- The model is tiny and fast: 250 vocabulary terms, 2,259 parameters, 23.3 KB on disk, 39 ms to train, single-request median latency 0.144 ms (P95 0.173 ms) on CPU.

## 2. Dataset

3,000 synthetic utterances (sheet `Training Data`): 9 supported intents × 300 (240 train / 30 validation / 30 test each) and 300 `UNKNOWN_INTENT` rows reserved for rejection evaluation (`evaluation_only`). Eight intents are `core`; `CHECK_SPENDING` is the `stretch` intent. The supplied splits were used unchanged. Figure: `results/figures/12_class_distribution.png`.

## 3. Data-quality findings

- No missing values, empty texts, invalid labels or contradictory labels.
- Heavy repetition: 1,004 redundant rows after normalization; only 179 distinct phrases among 270 test rows.
- Leakage: 21 test rows and 3 validation rows repeat a training phrase (exactly or after masking numbers). Reported, not removed; a leakage-controlled subset is evaluated.
- Strong vocabulary shift between train and held-out partitions (see §13).
Full details: `results/reports/data_quality_report.md`, `results/metrics/dataset_audit.{json,csv}`.

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

Grid: n-gram ∈ {(1,1), (1,2)} × C ∈ {0.1, 1, 10} × {counts, binary} = 12 configurations (`results/metrics/hyperparameter_search.csv`).

| rank | params_id | train_macro_f1 | validation_macro_f1 | validation_accuracy | validation_log_loss | vocabulary_size |
|---|---|---|---|---|---|---|
| 1 | ngram11_C0.1_count | 0.987 | 0.735 | 0.756 | 0.975 | 250 |
| 2 | ngram12_C0.1_binary | 0.993 | 0.732 | 0.744 | 1.017 | 1,739 |
| 3 | ngram12_C0.1_count | 0.993 | 0.732 | 0.744 | 1.017 | 1,739 |
| 4 | ngram12_C1_count | 1.000 | 0.731 | 0.744 | 0.785 | 1,739 |
| 5 | ngram12_C1_binary | 1.000 | 0.731 | 0.744 | 0.789 | 1,739 |
| 6 | ngram11_C0.1_binary | 0.987 | 0.723 | 0.741 | 0.974 | 250 |
| 7 | ngram12_C10_binary | 1.000 | 0.720 | 0.733 | 0.740 | 1,739 |
| 8 | ngram12_C10_count | 1.000 | 0.720 | 0.733 | 0.732 | 1,739 |
| 9 | ngram11_C10_count | 1.000 | 0.682 | 0.704 | 0.726 | 250 |
| 10 | ngram11_C10_binary | 1.000 | 0.680 | 0.704 | 0.705 | 250 |
| 11 | ngram11_C1_count | 1.000 | 0.673 | 0.700 | 0.724 | 250 |
| 12 | ngram11_C1_binary | 1.000 | 0.664 | 0.689 | 0.730 | 250 |

Selected: `ngram(1,1) C=0.1 counts`. Validation macro F1 spans only 0.664–0.735 across the grid. Best validation macro F1 by C: C=0.1: 0.735, C=1: 0.731, C=10: 0.720. With 30 validation rows per class, the gap between the top configurations is a few repeated phrases.

**Default vs tuned** (default = unigram counts, C = 1):

| model | train_F1 | val_F1 | test_acc | test_F1 | test_F1_leak_ctrl | test_ROC_AUC | test_log_loss |
|---|---|---|---|---|---|---|---|
| default (ngram(1,1) C=1 counts) | 1.000 | 0.673 | 0.852 | 0.855 | 0.833 | 0.965 | 0.696 |
| tuned (ngram(1,1) C=0.1 counts) | 0.987 | 0.735 | 0.793 | 0.796 | 0.774 | 0.954 | 0.985 |

Phrase-clustered bootstrap on test (cluster bootstrap over 179 groups, 2000 resamples, 95% percentile intervals): tuned macro F1 0.796 [0.712, 0.864], default 0.855 [0.774, 0.917]; difference -0.058 [-0.109, -0.012], positive in 0.5% of resamples. The default model's better test score is a held-out observation. It must not be used to switch models after the fact; doing so would turn the test set into a second validation set.

## 7. In-sample vs out-of-sample performance (selected model)

| split | n | accuracy | balanced_acc | macro_P | macro_R | macro_F1 | weighted_F1 | log_loss | macro_ROC_AUC | weighted_ROC_AUC | macro_AP | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train (in-sample) | 2,160 | 0.987 | 0.987 | 0.988 | 0.987 | 0.987 | 0.987 | 0.329 | 1.000 | 1.000 | 0.999 | 0.243 |
| validation (used for selection) | 270 | 0.756 | 0.756 | 0.740 | 0.756 | 0.735 | 0.735 | 0.975 | 0.957 | 0.957 | 0.795 | 0.222 |
| test (held-out) | 270 | 0.793 | 0.793 | 0.821 | 0.793 | 0.796 | 0.796 | 0.985 | 0.954 | 0.954 | 0.862 | 0.300 |
| test, leakage-controlled | 249 | 0.775 | 0.776 | 0.795 | 0.776 | 0.774 | 0.782 | 1.038 | 0.941 | 0.947 | 0.829 | 0.300 |
| test, one row per phrase | 179 | 0.832 | 0.828 | 0.826 | 0.828 | 0.814 | 0.841 | 0.866 | 0.963 | 0.969 | 0.887 | 0.301 |

Figure: `04_split_comparison.png`. The learning curve (`07_learning_curve.png`, training split only, 5-fold CV grouped by phrase) reaches 0.956 CV macro F1 at 1728 examples, far above validation (0.735). Phrases resembling training phrases are easy; the gap comes from new phrasings, which more rows of the same templates are unlikely to fix.

## 8. Per-class performance (test, selected model)

| intent | scope | precision | recall | f1 | support | roc_auc_ovr | average_precision |
|---|---|---|---|---|---|---|---|
| TRANSFER_MONEY | core | 1.000 | 0.667 | 0.800 | 30 | 0.962 | 0.849 |
| CHECK_BALANCE | core | 0.833 | 1.000 | 0.909 | 30 | 0.999 | 0.988 |
| VIEW_TRANSACTIONS | core | 0.553 | 0.700 | 0.618 | 30 | 0.828 | 0.586 |
| WITHDRAW_MONEY | core | 0.929 | 0.867 | 0.897 | 30 | 0.983 | 0.933 |
| BILL_PAY | core | 0.833 | 1.000 | 0.909 | 30 | 1.000 | 1.000 |
| HELP_SUPPORT | core | 0.545 | 0.600 | 0.571 | 30 | 0.928 | 0.712 |
| MANAGE_CONTACTS | core | 0.697 | 0.767 | 0.730 | 30 | 0.944 | 0.844 |
| VIEW_GROUP | core | 1.000 | 0.833 | 0.909 | 30 | 0.997 | 0.976 |
| CHECK_SPENDING | stretch | 1.000 | 0.700 | 0.824 | 30 | 0.945 | 0.874 |

**Core-eight breakdown** (9-class model, rows whose true label is core; predictions of CHECK_SPENDING count as errors):

| split | n_samples | accuracy | macro_precision | macro_recall | macro_f1 | predicted_outside_core |
|---|---|---|---|---|---|---|
| train | 1,920 | 0.985 | 0.989 | 0.985 | 0.987 | 5 |
| validation | 240 | 0.725 | 0.715 | 0.725 | 0.706 | 2 |
| test | 240 | 0.804 | 0.819 | 0.804 | 0.804 | 0 |

Supplementary 8-class model trained on core intents only (same hyperparameters, not re-tuned): test accuracy 0.808, macro F1 0.808, ROC-AUC 0.961 (`metrics_core8_model.json`).

## 9. Confusion-matrix analysis

Figures `01_confusion_matrix_counts.png` and `02_confusion_matrix_normalized.png`. Most frequent test confusions:

| actual_intent | predicted_intent | count | unique_phrases | example |
|---|---|---|---|---|
| TRANSFER_MONEY | HELP_SUPPORT | 8 | 8 | I owe joeli 100 |
| MANAGE_CONTACTS | VIEW_TRANSACTIONS | 7 | 5 | View my recipient list? |
| CHECK_SPENDING | VIEW_TRANSACTIONS | 6 | 4 | Show expenditure summary |
| HELP_SUPPORT | BILL_PAY | 6 | 3 | I have a complaint |
| HELP_SUPPORT | MANAGE_CONTACTS | 6 | 2 | Need customer service |
| VIEW_TRANSACTIONS | CHECK_BALANCE | 6 | 4 | Where is my statement |

Each confusion comes from one or a few templates (e.g. `I owe <name> <amount>`), often repeated several times, so each reflects a missing template, not a diffuse weakness. Common pattern: a key verb or noun absent from training (`owe`, `beneficiaries`, `statement`, `complaint`, `customer service`). The model then falls back to whatever weak cues remain, such as `show`, `view` or `my`.

## 10. ROC-AUC and precision-recall analysis

Test macro one-vs-rest ROC-AUC 0.954 (weighted 0.954); macro average precision 0.862. ROC-AUC is high relative to accuracy because the correct intent is usually ranked near the top even when it is not top-1: top-2 accuracy is 0.870 and top-3 0.904, vs top-1 0.793. A clarification prompt offering the top two intents is a cheap mitigation. Per-class curves: `05_roc_curves.png`, `06_precision_recall_curves.png`; values in `roc_auc_per_class.csv`. Undefined metrics in this run: none.

**Calibration:** test ECE 0.300 (`09_calibration_curve.png`). Mean top-label confidence is 0.493 against accuracy 0.793: the C = 0.1 model is **under-confident**. Its probabilities are flattened by regularization, which makes the absolute value of a confidence threshold hard to interpret.

## 11. Unknown-intent rejection

**Protocol.** Rule: `max_probability < threshold → UNKNOWN_INTENT`. The coverage target (95% of known validation intents accepted) was fixed before any unknown label was examined, giving threshold **0.1957** (validation coverage achieved 95.2%). The frozen threshold was applied once to test + the 300 evaluation-only unknowns.

| outcome | count |
|---|---|
| correct supported classification | 204 |
| incorrect supported classification | 41 |
| false rejection of supported | 25 |
| correct rejection of unsupported | 44 |
| false acceptance of unsupported | 256 |

- Known-intent coverage (test): **90.7%**; false rejection rate 9.3%; accuracy on accepted known requests 83.3%; end-to-end known accuracy 75.6%.
- Unknown rejection rate: **14.7%**; false acceptance rate **85.3%**. Per unique unknown phrase: 8.5% (153 phrases).
- Unknowns accepted, by predicted intent: VIEW_TRANSACTIONS 86, HELP_SUPPORT 69, CHECK_SPENDING 32, CHECK_BALANCE 16, BILL_PAY 15, TRANSFER_MONEY 14, MANAGE_CONTACTS 13, VIEW_GROUP 11.

The threshold-sensitivity curve (`10_rejection_threshold.png`, `rejection_threshold_sweep_exploratory.csv`) is **exploratory**: it re-uses the same evaluation-only unknowns at every threshold and is not an independent test.

**Supplementary unknown-aware threshold.** The 300 unknowns were split by normalized phrase into a calibration half (137 rows) and a disjoint holdout (163 rows, 0 shared phrases). Maximizing ½·coverage + ½·rejection on validation knowns + calibration unknowns selects threshold 0.41. On test knowns + holdout unknowns: unknown rejection 94.5% but known coverage only 61.1%. The primary threshold on the same holdout gives 22.1% / 90.7%. Confidence alone cannot separate known from unknown requests at an acceptable operating point. Caveat: both halves come from the same synthetic generator, and the holdout was also part of the primary evaluation, so this is not a fully untouched test.

## 12. Computational performance

| Measure | Value |
|---|---|
| Training time (final fit, 2,160 rows) | 38.6 ms |
| Full grid search (12 fits + scoring) | 1.13 s |
| Single-request latency median / P95 / P99 | 0.144 / 0.173 / 0.280 ms |
| Single-request throughput (1 / median) | 6,944 req/s |
| Batch of 270 latency median / P95 | 0.97 / 2.03 ms |
| Batch throughput | 277,079 predictions/s |
| Serialized model (joblib bundle) | 23.3 KB |
| Vocabulary size | 250 |
| Parameters (coefficients + intercepts) | 2,259 (9×250 + 9) |
| Hardware | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 32 logical CPUs, 31.73 GB RAM, CPU only |
| Runtime | Windows-11-10.0.26200-SP0, Python 3.13.15, scikit-learn 1.9.1 |

Methodology: time.perf_counter around one end-to-end predict_proba call on raw text (normalization + vectorization + LR). Warm-up calls discarded. Single-request batches sample random test utterances; batch measurement uses all 270 test rows. Single process, no other workload controls; wall-clock times vary by machine. Single-request: 50 warm-up + 1000 measured calls; batch: 10 warm-up + 200 measured calls. Figure `11_inference_latency.png`.

## 13. Error analysis (summary)

| split | n | short_n | short_accuracy | longer_accuracy | rows_with_oov | share_rows_with_oov | oov_accuracy | in_vocab_accuracy | mean_oov_token_share |
|---|---|---|---|---|---|---|---|---|---|
| train | 2,160 | 178 | 1.000 | 0.986 | 0 | 0.000 | n/a | 0.987 | 0.000 |
| validation | 270 | 15 | 0.867 | 0.749 | 88 | 0.326 | 0.591 | 0.835 | 0.105 |
| test | 270 | 44 | 0.841 | 0.783 | 125 | 0.463 | 0.616 | 0.945 | 0.136 |

- 78 distinct validation/test phrases are misclassified **and** accepted by the rejection threshold. 0 held-out errors have confidence ≥ 0.9, because the regularized model rarely produces high confidence.
- Word order: BoW gives identical probabilities to reordered commands (`transfer from mere to jone` vs `transfer from jone to mere`). Direction of payment is invisible to this model.
- Hand-written diagnostic probes are qualitative only (see `error_analysis.md`).

Full analysis: `results/reports/error_analysis.md`.

## 14. Limitations and potential leakage

- **Synthetic data.** Utterances are generated from templates with filler suffixes. Real customer language in Fiji will include Fijian and Fiji Hindi words, code-switching, local slang and abbreviations (e.g. M-PAiSA, "vinaka", "bhaiya"), misspellings, SMS shorthand and speech-to-text errors. None of these is represented systematically here. Reported metrics should not be read as production accuracy.
- **Leakage.** 21 test rows repeat a training phrase. Leakage-controlled test macro F1 is 0.774 vs 0.796 on the full test set.
- **Small, repetitive held-out sets.** 179 distinct test phrases; confidence intervals are wide (see §6), so small differences in macro F1 between models are not reliable.
- **Unknown evaluation.** The 300 unknowns are synthetic, highly repetitive and sometimes padded with wallet words ("… on my account", "… on the app"). Rejection results are specific to this set.
- **BoW limits.** No word order, negation, synonyms or sub-word information. Unseen words are dropped silently.
- **Calibration.** Probabilities are under-confident; threshold values do not transfer across models without recalibration.

## 15. Recommendations for the next model

1. **Sub-word / character features** (char n-gram TF-IDF, fastText or a transformer tokenizer) to handle unseen words, misspellings and SMS forms. This is the dominant failure mode here.
2. **Pretrained sentence encoders or transformers** to map unseen paraphrases (`owe`, `beneficiaries`, `statement`) to known intents through semantic similarity.
3. **A dedicated out-of-scope detector** (e.g. trained on a separate pool of unsupported requests, energy/Mahalanobis scores or an embedding-distance check). Keep a genuinely untouched unknown test set.
4. **Calibrate probabilities** (temperature scaling on validation) before setting thresholds, and select thresholds on a separate calibration split.
5. **Safety confirmation for money-moving intents.** Require explicit confirmation of amount and payee for TRANSFER_MONEY, BILL_PAY and WITHDRAW_MONEY regardless of model confidence.
6. **Data:** collect or write more diverse paraphrases per intent, de-duplicate phrases, and use phrase-grouped splits so validation reflects test difficulty.
7. **Reuse this evaluation framework** (`sole_intent.evaluate`, `sole_intent.rejection`) and append to `experiment_summary.csv` so all models are compared on identical splits and metrics.
