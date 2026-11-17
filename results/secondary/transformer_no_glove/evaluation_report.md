# Custom Transformer - evaluation report

> **SECONDARY EXPERIMENT (no_glove)** - trained on train; not part of the primary four-model comparison.

_Generated 2026-11-16T18:06:51+00:00 from the model's actual predictions. Dataset `v5_deduplicated` (SHA-256 `8dc9c89cbb8af41f...`). Original implementation: Felipe (friosf)._

## Headline (raw classifier, no confidence threshold)

| Metric | Value |
|---|---|
| Raw test accuracy | **94.4%** (255 / 270) |
| Test macro precision / recall / F1 | 0.947 / 0.944 / **0.944** |
| Test weighted F1 | 0.944 |
| Test macro ROC-AUC (one-vs-rest) | 0.996 |
| Train / validation accuracy | 99.9% / 97.0% |
| Parameters | 232,561 |
| Serialized model size | 2.93 MB |
| Training time (final fit / incl. search) | 69.6 s / 72.1 s |
| Latency P50 / P95 (single command, cuda) | 13.60 / 16.28 ms |

Raw accuracy counts every one of the legitimate test commands, before any confidence threshold.

## Selected configuration (chosen on validation only)

- Hyperparameters: `{'num_heads': 4, 'num_blocks': 2, 'dropout': 0.2, 'learning_rate': 0.001, 'pretrained': False, 'epochs': 16}`
- Rule: highest validation accuracy; ties -> lowest validation log loss; best epoch kept (patience 10)
- Configurations searched: 1

## Per-intent results (test)

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| TRANSFER_MONEY | 0.967 | 0.967 | 0.967 | 30 |
| CHECK_BALANCE | 0.968 | 1.000 | 0.984 | 30 |
| VIEW_TRANSACTIONS | 0.844 | 0.900 | 0.871 | 30 |
| WITHDRAW_MONEY | 0.882 | 1.000 | 0.938 | 30 |
| BILL_PAY | 1.000 | 1.000 | 1.000 | 30 |
| HELP_SUPPORT | 0.960 | 0.800 | 0.873 | 30 |
| MANAGE_CONTACTS | 0.931 | 0.900 | 0.915 | 30 |
| VIEW_GROUP | 1.000 | 0.933 | 0.966 | 30 |
| CHECK_SPENDING | 0.968 | 1.000 | 0.984 | 30 |

## Confidence threshold

Default threshold **0.9891**, calibrated on validation only: largest observed validation max-probability with coverage >= target (sole_intent.rejection.threshold_for_coverage); target coverage 95%, actual validation coverage 95.2%. No test or unknown-evaluation labels were used.

| Threshold | Coverage | Accepted | Correct accepted | Accuracy among accepted | Overall correct routing | Rejected | Unknown rejected | Unknown -> HELP_SUPPORT | Unknown safe fallback | Unknown -> money flow |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 100.0% | 270 | 255 | 94.4% | 94.4% | 0 | 0.0% | 28 | 9.3% | 34 |
| default (0.989) | 94.4% | 255 | 248 | 97.3% | 91.9% | 15 | 50.3% | 11 | 54.0% | 25 |
| 0.25 | 100.0% | 270 | 255 | 94.4% | 94.4% | 0 | 0.0% | 28 | 9.3% | 34 |
| 0.30 | 100.0% | 270 | 255 | 94.4% | 94.4% | 0 | 0.0% | 28 | 9.3% | 34 |
| 0.35 | 100.0% | 270 | 255 | 94.4% | 94.4% | 0 | 0.0% | 28 | 9.3% | 34 |
| 0.40 | 100.0% | 270 | 255 | 94.4% | 94.4% | 0 | 0.0% | 28 | 9.3% | 34 |
| 0.50 | 99.6% | 269 | 255 | 94.8% | 94.4% | 1 | 1.0% | 26 | 9.7% | 34 |
| 0.60 | 99.6% | 269 | 255 | 94.8% | 94.4% | 1 | 5.0% | 24 | 13.0% | 33 |

*Accuracy among accepted* uses only the accepted commands as denominator; *overall correct routing* uses all legitimate test commands and counts rejections as failures. UNKNOWN_INTENT (rejected) and HELP_SUPPORT are counted separately; both are safe fallbacks.

## Secondary analyses (not part of the primary comparison)

- Typo robustness: accuracy with one keyboard slip per command 79.6% (clean 94.4%), macro F1 0.801.
- Temperature scaling (fit on validation): T = 1.986; test ECE 0.052 -> 0.034, test log loss 0.492 -> 0.274.
- Three-tier routing (open >= 0.989, menu >= 0.709): 91.9% of legitimate commands open the right workflow, 3.3% get a top-3 menu containing the right intent, 2.6% open a wrong workflow; 49.7% of unknown commands open a workflow (8.3% a money flow).

## Figures

![Confusion matrix](confusion_matrix.png)

![ROC curves](roc_curves.png)

![Threshold trade-off](threshold_sweep.png)

## Provenance

- Training command: `python train.py --model transformer --variant no_glove`; evaluation command: `python evaluate.py --model transformer --label no_glove`
- Trained on: train; selected on: validation
- Artifact: `artifacts/secondary/transformer_no_glove` (5 files, hashes in `experiment_metadata.json`)
- Git commit at evaluation: `1cd668e6e692a9b99e4b4bd20a3bff4c7a08faf0`
