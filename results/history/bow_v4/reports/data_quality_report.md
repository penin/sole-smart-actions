# Data Quality Report — Sole Smart Actions Intent Dataset

_Generated 2026-11-09T04:20:40+00:00 by `scripts/run_all.py`. Source workbook is read-only; no rows were deleted, edited or re-split._

## Source

- File: `Sole_Smart_Actions_Intent_Training_v4_Cleaned.xlsx` (sheet `Training Data`)
- Dataset version: `v4_cleaned`; rows: 3,000
- File SHA-256: `708e8f79ca47c0df689c97da02b21017607e2f55e50b30dad4cb0d9cc7033bda`
- Content SHA-256 (text,intent,split,scope as CSV): `4800e7a43c1c34f4ddf206c2833df6c91f6e63ca1444c80653742aa3b14180c0`

## Integrity checks

| category | check | value | status | details |
|---|---|---|---|---|
| schema | row_count | 3000.000 | ok | expected 3000 |
| schema | missing_text | 0.000 | ok |  |
| schema | missing_intent | 0.000 | ok |  |
| schema | missing_split | 0.000 | ok |  |
| schema | missing_scope | 0.000 | ok |  |
| schema | empty_or_whitespace_text | 0.000 | ok |  |
| schema | invalid_intent_labels | 0.000 | ok |  |
| schema | invalid_split_values | 0.000 | ok |  |
| schema | scope_intent_mismatch | 0.000 | ok | core -> 8 core intents; stretch -> CHECK_SPENDING; rejection_test -> UNKNOWN_INTENT |
| schema | unknown_rows_outside_evaluation_only | 0.000 | ok |  |
| splits | count_train | 2160.000 | ok | expected 2160 |
| splits | count_validation | 270.000 | ok | expected 270 |
| splits | count_test | 270.000 | ok | expected 270 |
| splits | count_evaluation_only | 300.000 | ok | expected 300 |
| labels | contradictory_label_phrases | 0.000 | ok | same normalized phrase, different intent |
| labels | contradictory_label_templates | 0.000 | ok | same number-masked template, different intent |

## Class balance by split

| index | train | validation | test | evaluation_only |
|---|---|---|---|---|
| TRANSFER_MONEY | 240 | 30 | 30 | 0 |
| CHECK_BALANCE | 240 | 30 | 30 | 0 |
| VIEW_TRANSACTIONS | 240 | 30 | 30 | 0 |
| WITHDRAW_MONEY | 240 | 30 | 30 | 0 |
| BILL_PAY | 240 | 30 | 30 | 0 |
| HELP_SUPPORT | 240 | 30 | 30 | 0 |
| MANAGE_CONTACTS | 240 | 30 | 30 | 0 |
| VIEW_GROUP | 240 | 30 | 30 | 0 |
| CHECK_SPENDING | 240 | 30 | 30 | 0 |
| UNKNOWN_INTENT | 0 | 0 | 0 | 300 |

Supported intents are perfectly balanced (240 / 30 / 30). `CHECK_SPENDING` is the stretch intent (`scope = stretch`); the other eight are `core`. All 300 `UNKNOWN_INTENT` rows are `evaluation_only` / `rejection_test` and are never used as a training class.

## Utterance length

Tokens per utterance: mean 4.62, median 4, min 1, max 12. Utterances are short commands, so a single unseen word can remove most of the usable signal.

| split | intent | n_tokens_mean | n_tokens_min | n_tokens_max |
|---|---|---|---|---|
| test | BILL_PAY | 3.27 | 2 | 6 |
| test | CHECK_BALANCE | 4.03 | 2 | 7 |
| test | CHECK_SPENDING | 3.97 | 2 | 7 |
| test | HELP_SUPPORT | 4.10 | 1 | 8 |
| test | MANAGE_CONTACTS | 3.73 | 2 | 5 |
| test | TRANSFER_MONEY | 4.60 | 4 | 6 |
| test | VIEW_GROUP | 4.87 | 2 | 8 |
| test | VIEW_TRANSACTIONS | 3.43 | 2 | 7 |
| test | WITHDRAW_MONEY | 3.47 | 3 | 6 |
| train | BILL_PAY | 4.05 | 2 | 10 |
| train | CHECK_BALANCE | 5.90 | 1 | 10 |
| train | CHECK_SPENDING | 6.38 | 2 | 12 |
| train | HELP_SUPPORT | 4.52 | 2 | 11 |
| train | MANAGE_CONTACTS | 3.65 | 2 | 5 |
| train | TRANSFER_MONEY | 5.33 | 3 | 8 |
| train | VIEW_GROUP | 4.26 | 2 | 10 |
| train | VIEW_TRANSACTIONS | 4.70 | 2 | 10 |
| train | WITHDRAW_MONEY | 4.21 | 2 | 8 |
| validation | BILL_PAY | 4.73 | 4 | 7 |
| validation | CHECK_BALANCE | 4.70 | 4 | 7 |
| validation | CHECK_SPENDING | 3.50 | 1 | 6 |
| validation | HELP_SUPPORT | 4.73 | 4 | 7 |
| validation | MANAGE_CONTACTS | 2.87 | 2 | 3 |
| validation | TRANSFER_MONEY | 4.20 | 3 | 6 |
| validation | VIEW_GROUP | 4.90 | 4 | 7 |
| validation | VIEW_TRANSACTIONS | 4.73 | 2 | 8 |
| validation | WITHDRAW_MONEY | 2.93 | 2 | 4 |

## Duplicates

| category | check | value | status | details |
|---|---|---|---|---|
| duplicates | exact_duplicate_rows | 1148.000 | warn | 723 redundant copies |
| duplicates | normalized_duplicate_rows | 1503.000 | warn | 1004 redundant copies; 499 repeated phrases |
| duplicates | number_masked_duplicate_rows | 1911.000 | info | same phrase after masking numbers (e.g. 'send 50' vs 'send 500') |
| duplicates | within_split_redundant_train | 693.000 | warn | 1467 unique of 2160 |
| duplicates | within_split_redundant_validation | 60.000 | warn | 210 unique of 270 |
| duplicates | within_split_redundant_test | 91.000 | warn | 179 unique of 270 |
| duplicates | within_split_redundant_evaluation_only | 147.000 | warn | 153 unique of 300 |

Only 1,996 of 3,000 rows are distinct after normalization (lowercase, punctuation-insensitive). Unique phrases per partition: train 1,467, validation 210, test 179, evaluation_only 153. Repeats are exact copies or differ only in case/punctuation (e.g. a trailing `?`). Within-partition repetition inflates the apparent sample size: the 270-row test set holds 179 distinct phrases, so test confidence intervals use a phrase-clustered bootstrap.

Most repeated phrases:

| canonical | count | splits | intents |
|---|---|---|---|
| app keeps crashing | 12 | train | HELP_SUPPORT |
| play music | 11 | evaluation_only | UNKNOWN_INTENT |
| recent transactions | 9 | test | VIEW_TRANSACTIONS |
| forgot password | 9 | train | HELP_SUPPORT |
| reset my pin | 9 | train | HELP_SUPPORT |
| how tall is mount everest | 9 | evaluation_only | UNKNOWN_INTENT |
| where is the nearest beach | 9 | evaluation_only | UNKNOWN_INTENT |
| i forgot my pin | 9 | train | HELP_SUPPORT |
| predict the lottery | 8 | evaluation_only | UNKNOWN_INTENT |
| talk to an agent | 8 | validation | HELP_SUPPORT |
| report fraud | 8 | train | HELP_SUPPORT |
| open youtube | 8 | evaluation_only | UNKNOWN_INTENT |
| i can t access my account | 8 | train | HELP_SUPPORT |
| transfer failed | 8 | train | HELP_SUPPORT |
| tell me a joke | 8 | evaluation_only | UNKNOWN_INTENT |

## Contradictory labels

0 normalized phrases carry more than one intent label — no label conflicts were found.

## Cross-partition overlap and leakage

| category | check | value | status | details |
|---|---|---|---|---|
| leakage | shared_phrases_train_vs_validation | 2.000 | warn | 3 validation rows; 2 shared number-masked templates |
| leakage | shared_phrases_train_vs_test | 11.000 | fail | 16 test rows; 13 shared number-masked templates |
| leakage | shared_phrases_train_vs_evaluation_only | 0.000 | ok | 0 evaluation_only rows; 0 shared number-masked templates |
| leakage | shared_phrases_validation_vs_test | 0.000 | ok | 0 test rows; 0 shared number-masked templates |
| leakage | shared_phrases_validation_vs_evaluation_only | 0.000 | ok | 0 evaluation_only rows; 0 shared number-masked templates |
| leakage | shared_phrases_test_vs_evaluation_only | 0.000 | ok | 0 evaluation_only rows; 0 shared number-masked templates |
| leakage | validation_rows_normalized_in_train | 3.000 | warn |  |
| leakage | validation_rows_template_in_train | 3.000 | warn |  |
| leakage | validation_rows_near_duplicate_of_train | 4.000 | info | token Jaccard >= 0.8; median max-Jaccard 0.50 |
| leakage | test_rows_normalized_in_train | 16.000 | fail |  |
| leakage | test_rows_template_in_train | 21.000 | warn |  |
| leakage | test_rows_near_duplicate_of_train | 17.000 | info | token Jaccard >= 0.8; median max-Jaccard 0.44 |
| leakage | evaluation_only_rows_normalized_in_train | 0.000 | ok |  |
| leakage | evaluation_only_rows_template_in_train | 0.000 | ok |  |
| leakage | evaluation_only_rows_near_duplicate_of_train | 0.000 | info | token Jaccard >= 0.8; median max-Jaccard 0.25 |

| Partition | Rows | Phrase seen in train | Number-masked template seen in train | Near-duplicate (Jaccard ≥ 0.8) | Leakage-flagged |
|---|---|---|---|---|---|
| validation | 270 | 3 | 3 | 4 | 3 |
| test | 270 | 16 | 21 | 17 | 21 |
| evaluation_only | 300 | 0 | 0 | 0 | 0 |

**Leakage flag:** 21 test rows (7.8%) repeat a training phrase exactly or after masking numbers. The primary benchmark keeps the supplied split; a **leakage-controlled test subset** excluding these rows is evaluated separately (`metrics_test_leakage_controlled.json`). Unknown-evaluation phrases do not overlap training.

Rows flagged per partition are listed in `data/processed/leakage_rows.csv`; all shared phrases in `data/processed/cross_split_overlap.csv`.

## Vocabulary shift (train vocabulary vs held-out partitions)

| split | n | rows_with_oov | share_rows_with_oov | mean_oov_token_share |
|---|---|---|---|---|
| train | 2,160 | 0 | 0.000 | 0.000 |
| validation | 270 | 88 | 0.326 | 0.105 |
| test | 270 | 125 | 0.463 | 0.136 |

Validation and test contain many words that never occur in training (e.g. `edit`, `agent`, `has`, `received`, `talk`, `with`, `recent`, `customer`, `service`, `purchases`, `look`, `available`). The held-out partitions therefore test generalization to **unseen paraphrase templates**, not just unseen rows. This is the main driver of the train/validation gap reported in the evaluation report.

## Implications

1. Training-set metrics and cross-validation inside the training split are optimistic (duplicates and shared templates).
2. Validation/test numbers are dominated by a small number of unseen templates repeated several times; single errors move metrics in steps of several points.
3. Exact train→test overlap is small but non-zero and is reported, not removed.
4. Many unknown-evaluation phrases repeat (e.g. `play music` ×11); unknown rejection is also reported per unique phrase.
