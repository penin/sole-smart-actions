# Custom Transformer - evaluation report

> **SECONDARY EXPERIMENT (no_attention)** - trained on train; not part of the primary four-model comparison.

_Generated 2026-11-16T18:05:13+00:00 from the model's actual predictions. Dataset `v5_deduplicated` (SHA-256 `8dc9c89cbb8af41f...`). Original implementation: Felipe (friosf)._

## Headline (raw classifier, no confidence threshold)

| Metric | Value |
|---|---|
| Raw test accuracy | **95.6%** (258 / 270) |
| Test macro precision / recall / F1 | 0.957 / 0.956 / **0.956** |
| Test weighted F1 | 0.956 |
| Test macro ROC-AUC (one-vs-rest) | 0.997 |
| Train / validation accuracy | 100.0% / 89.3% |
| Parameters | 47,849 |
| Serialized model size | 0.62 MB |
| Training time (final fit / incl. search) | 24.5 s / 26.6 s |
| Latency P50 / P95 (single command, cuda) | 3.64 / 7.31 ms |

Raw accuracy counts every one of the legitimate test commands, before any confidence threshold.

## Selected configuration (chosen on validation only)

- Hyperparameters: `{'num_heads': 4, 'num_blocks': 0, 'dropout': 0.2, 'learning_rate': 0.001, 'epochs': 16}`
- Rule: highest validation accuracy; ties -> lowest validation log loss; best epoch kept (patience 10)
- Configurations searched: 1

## Per-intent results (test)

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| TRANSFER_MONEY | 0.964 | 0.900 | 0.931 | 30 |
| CHECK_BALANCE | 0.882 | 1.000 | 0.938 | 30 |
| VIEW_TRANSACTIONS | 0.931 | 0.900 | 0.915 | 30 |
| WITHDRAW_MONEY | 0.903 | 0.933 | 0.918 | 30 |
| BILL_PAY | 1.000 | 1.000 | 1.000 | 30 |
| HELP_SUPPORT | 0.935 | 0.967 | 0.951 | 30 |
| MANAGE_CONTACTS | 1.000 | 0.933 | 0.966 | 30 |
| VIEW_GROUP | 1.000 | 1.000 | 1.000 | 30 |
| CHECK_SPENDING | 1.000 | 0.967 | 0.983 | 30 |

## Confidence threshold

Default threshold **0.5389**, calibrated on validation only: largest observed validation max-probability with coverage >= target (sole_intent.rejection.threshold_for_coverage); target coverage 95%, actual validation coverage 95.2%. No test or unknown-evaluation labels were used.

| Threshold | Coverage | Accepted | Correct accepted | Accuracy among accepted | Overall correct routing | Rejected | Unknown rejected | Unknown -> HELP_SUPPORT | Unknown safe fallback | Unknown -> money flow |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 100.0% | 270 | 258 | 95.6% | 95.6% | 0 | 0.0% | 131 | 43.7% | 14 |
| default (0.539) | 97.4% | 263 | 256 | 97.3% | 94.8% | 7 | 35.7% | 102 | 69.7% | 10 |
| 0.25 | 100.0% | 270 | 258 | 95.6% | 95.6% | 0 | 0.0% | 131 | 43.7% | 14 |
| 0.30 | 99.3% | 268 | 258 | 96.3% | 95.6% | 2 | 1.0% | 128 | 43.7% | 14 |
| 0.35 | 99.3% | 268 | 258 | 96.3% | 95.6% | 2 | 7.7% | 126 | 49.7% | 12 |
| 0.40 | 99.3% | 268 | 258 | 96.3% | 95.6% | 2 | 14.7% | 124 | 56.0% | 11 |
| 0.50 | 97.8% | 264 | 256 | 97.0% | 94.8% | 6 | 32.0% | 108 | 68.0% | 10 |
| 0.60 | 96.7% | 261 | 254 | 97.3% | 94.1% | 9 | 50.3% | 77 | 76.0% | 10 |

*Accuracy among accepted* uses only the accepted commands as denominator; *overall correct routing* uses all legitimate test commands and counts rejections as failures. UNKNOWN_INTENT (rejected) and HELP_SUPPORT are counted separately; both are safe fallbacks.

## Secondary analyses (not part of the primary comparison)

- Typo robustness: accuracy with one keyboard slip per command 88.5% (clean 95.6%), macro F1 0.884.
- Temperature scaling (fit on validation): T = 1.445; test ECE 0.019 -> 0.041, test log loss 0.172 -> 0.185.
- Three-tier routing (open >= 0.539, menu >= 0.337): 94.8% of legitimate commands open the right workflow, 1.9% get a top-3 menu containing the right intent, 2.6% open a wrong workflow; 64.3% of unknown commands open a workflow (3.3% a money flow).

## Figures

![Confusion matrix](confusion_matrix.png)

![ROC curves](roc_curves.png)

![Threshold trade-off](threshold_sweep.png)

## Provenance

- Training command: `python train.py --model transformer --variant no_attention`; evaluation command: `python evaluate.py --model transformer --label no_attention`
- Trained on: train; selected on: validation
- Artifact: `artifacts/secondary/transformer_no_attention` (5 files, hashes in `experiment_metadata.json`)
- Git commit at evaluation: `1cd668e6e692a9b99e4b4bd20a3bff4c7a08faf0`
