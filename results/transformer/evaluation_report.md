# Custom Transformer - evaluation report


_Generated 2026-11-16T18:08:14+00:00 from the model's actual predictions. Dataset `v5_deduplicated` (SHA-256 `8dc9c89cbb8af41f...`). Original implementation: Felipe (friosf)._

## Headline (raw classifier, no confidence threshold)

| Metric | Value |
|---|---|
| Raw test accuracy | **97.4%** (263 / 270) |
| Test macro precision / recall / F1 | 0.977 / 0.974 / **0.974** |
| Test weighted F1 | 0.974 |
| Test macro ROC-AUC (one-vs-rest) | 1.000 |
| Train / validation accuracy | 99.9% / 93.0% |
| Parameters | 232,561 |
| Serialized model size | 2.93 MB |
| Training time (final fit / incl. search) | 36.2 s / 1640.8 s |
| Latency P50 / P95 (single command, cuda) | 13.62 / 20.87 ms |

Raw accuracy counts every one of the legitimate test commands, before any confidence threshold.

## Selected configuration (chosen on validation only)

- Hyperparameters: `{'num_heads': 4, 'num_blocks': 2, 'dropout': 0.2, 'learning_rate': 0.001, 'epochs': 5}`
- Rule: highest validation accuracy; ties -> lowest validation log loss; best epoch kept (patience 10)
- Configurations searched: 36

## Per-intent results (test)

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| TRANSFER_MONEY | 1.000 | 1.000 | 1.000 | 30 |
| CHECK_BALANCE | 0.882 | 1.000 | 0.938 | 30 |
| VIEW_TRANSACTIONS | 1.000 | 0.933 | 0.966 | 30 |
| WITHDRAW_MONEY | 1.000 | 1.000 | 1.000 | 30 |
| BILL_PAY | 1.000 | 1.000 | 1.000 | 30 |
| HELP_SUPPORT | 1.000 | 0.833 | 0.909 | 30 |
| MANAGE_CONTACTS | 0.909 | 1.000 | 0.952 | 30 |
| VIEW_GROUP | 1.000 | 1.000 | 1.000 | 30 |
| CHECK_SPENDING | 1.000 | 1.000 | 1.000 | 30 |

## Confidence threshold

Default threshold **0.6597**, calibrated on validation only: largest observed validation max-probability with coverage >= target (sole_intent.rejection.threshold_for_coverage); target coverage 95%, actual validation coverage 95.2%. No test or unknown-evaluation labels were used.

| Threshold | Coverage | Accepted | Correct accepted | Accuracy among accepted | Overall correct routing | Rejected | Unknown rejected | Unknown -> HELP_SUPPORT | Unknown safe fallback | Unknown -> money flow |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 100.0% | 270 | 263 | 97.4% | 97.4% | 0 | 0.0% | 32 | 10.7% | 35 |
| default (0.660) | 96.3% | 260 | 255 | 98.1% | 94.4% | 10 | 24.0% | 25 | 32.3% | 17 |
| 0.25 | 100.0% | 270 | 263 | 97.4% | 97.4% | 0 | 0.0% | 32 | 10.7% | 35 |
| 0.30 | 100.0% | 270 | 263 | 97.4% | 97.4% | 0 | 0.0% | 32 | 10.7% | 35 |
| 0.35 | 100.0% | 270 | 263 | 97.4% | 97.4% | 0 | 0.0% | 32 | 10.7% | 35 |
| 0.40 | 100.0% | 270 | 263 | 97.4% | 97.4% | 0 | 2.3% | 31 | 12.7% | 35 |
| 0.50 | 99.3% | 268 | 261 | 97.4% | 96.7% | 2 | 10.0% | 28 | 19.3% | 35 |
| 0.60 | 97.0% | 262 | 256 | 97.7% | 94.8% | 8 | 19.3% | 25 | 27.7% | 18 |

*Accuracy among accepted* uses only the accepted commands as denominator; *overall correct routing* uses all legitimate test commands and counts rejections as failures. UNKNOWN_INTENT (rejected) and HELP_SUPPORT are counted separately; both are safe fallbacks.

## Secondary analyses (not part of the primary comparison)

- Typo robustness: accuracy with one keyboard slip per command 78.1% (clean 97.4%), macro F1 0.787.
- Temperature scaling (fit on validation): T = 1.984; test ECE 0.027 -> 0.053, test log loss 0.094 -> 0.123.
- Three-tier routing (open >= 0.660, menu >= 0.523): 94.4% of legitimate commands open the right workflow, 3.0% get a top-3 menu containing the right intent, 1.9% open a wrong workflow; 76.0% of unknown commands open a workflow (5.7% a money flow).

## Figures

![Confusion matrix](confusion_matrix.png)

![ROC curves](roc_curves.png)

![Threshold trade-off](threshold_sweep.png)

## Provenance

- Training command: `python -c`; evaluation command: `python evaluate.py --model all`
- Trained on: train; selected on: validation
- Artifact: `artifacts/transformer` (5 files, hashes in `experiment_metadata.json`)
- Git commit at evaluation: `1cd668e6e692a9b99e4b4bd20a3bff4c7a08faf0`
