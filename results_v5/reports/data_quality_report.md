# Data Quality Report — Sole Smart Actions Intent Dataset

_Generated 2026-11-12T02:55:57+00:00 by `scripts/run_all.py`. Source workbook is read-only; no rows were deleted, edited or re-split._

## Source

- File: `Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx` (sheet `Training Data`)
- Dataset version: `v5_deduplicated`; rows: 3,000
- File SHA-256: `8dc9c89cbb8af41fd7cb25b2841e2f1a0c986641c4a6108c2f804d09f9345516`
- Content SHA-256 (text,intent,split,scope as CSV): `8277d2569ed2373b3b30062761c54d41ea50a63895c19716b34c35293eaffabe`

## Pre-training validation (required checks)

**Result: PASSED.** Checks run before any training. A failure stops the run and nothing in the dataset is changed. Duplicate key: Unicode NFKC, casefold, collapsed whitespace, surrounding punctuation removed. Rows marked `info` are stricter diagnostics, not requirements.

| check | value | expected | status | details |
|---|---|---|---|---|
| total_rows | 3,000 | 3,000 | pass |  |
| rows_train | 2,160 | 2,160 | pass |  |
| rows_validation | 270 | 270 | pass |  |
| rows_test | 270 | 270 | pass |  |
| rows_evaluation_only | 300 | 300 | pass |  |
| rows_per_intent_including_unknown | 300–300 | 300 | pass | all 10 labels |
| missing_command_text | 0 | 0 | pass |  |
| missing_intent_label | 0 | 0 | pass |  |
| exact_duplicate_commands | 0 | 0 | pass |  |
| normalized_duplicate_commands | 0 | 0 | pass | NFKC + casefold + whitespace + surrounding punctuation;  |
| cross_split_duplicate_commands | 0 | 0 | pass | normalized key shared by two splits |
| conflicting_labels_same_normalized_command | 0 | 0 | pass |  |
| unknown_intent_evaluation_only | 0 | 0 | pass | UNKNOWN_INTENT rows outside evaluation_only |
| info_duplicates_ignoring_interior_punctuation | 14 | info | info | How tall is mount everest? if possible; Spending this month? in sole; Forgot password? now; App keeps crashing on sole; App keeps crashing? now; App keeps crashing now |
| info_validation_number_masked_template_in_train | 23 | info | info | same command as a training row once numbers are masked |
| info_test_number_masked_template_in_train | 39 | info | info | same command as a training row once numbers are masked |
| info_evaluation_only_number_masked_template_in_train | 0 | info | info | same command as a training row once numbers are masked |
| info_replacement_log_rows | 997 | info | info | duplicate occurrences replaced by the dataset author (from the workbook) |

The workbook's *Replacement Log* records 997 duplicate occurrences that the dataset author replaced with new synthetic commands (by split: train 693, evaluation_only 146, test 97, validation 61).

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

Tokens per utterance: mean 5.78, median 6, min 1, max 12. Utterances are short commands, so a single unseen word can remove most of the usable signal.

| split | intent | n_tokens_mean | n_tokens_min | n_tokens_max |
|---|---|---|---|---|
| test | BILL_PAY | 5.40 | 2 | 8 |
| test | CHECK_BALANCE | 6.93 | 2 | 11 |
| test | CHECK_SPENDING | 5.10 | 2 | 10 |
| test | HELP_SUPPORT | 6.77 | 1 | 12 |
| test | MANAGE_CONTACTS | 4.43 | 2 | 7 |
| test | TRANSFER_MONEY | 4.60 | 4 | 6 |
| test | VIEW_GROUP | 5.53 | 2 | 8 |
| test | VIEW_TRANSACTIONS | 5.97 | 2 | 9 |
| test | WITHDRAW_MONEY | 4.17 | 3 | 8 |
| train | BILL_PAY | 5.45 | 2 | 10 |
| train | CHECK_BALANCE | 7.73 | 1 | 11 |
| train | CHECK_SPENDING | 7.09 | 2 | 12 |
| train | HELP_SUPPORT | 6.51 | 2 | 12 |
| train | MANAGE_CONTACTS | 4.01 | 2 | 7 |
| train | TRANSFER_MONEY | 5.40 | 3 | 8 |
| train | VIEW_GROUP | 6.31 | 2 | 11 |
| train | VIEW_TRANSACTIONS | 5.53 | 2 | 10 |
| train | WITHDRAW_MONEY | 4.76 | 2 | 8 |
| validation | BILL_PAY | 4.73 | 4 | 7 |
| validation | CHECK_BALANCE | 5.90 | 4 | 10 |
| validation | CHECK_SPENDING | 5.60 | 1 | 10 |
| validation | HELP_SUPPORT | 6.63 | 4 | 11 |
| validation | MANAGE_CONTACTS | 3.30 | 2 | 6 |
| validation | TRANSFER_MONEY | 4.20 | 3 | 6 |
| validation | VIEW_GROUP | 5.30 | 4 | 8 |
| validation | VIEW_TRANSACTIONS | 5.37 | 2 | 9 |
| validation | WITHDRAW_MONEY | 4.03 | 2 | 7 |

## Duplicates

| category | check | value | status | details |
|---|---|---|---|---|
| duplicates | exact_duplicate_rows | 0.000 | ok | 0 redundant copies |
| duplicates | normalized_duplicate_rows | 14.000 | warn | 7 redundant copies; 7 repeated phrases |
| duplicates | number_masked_duplicate_rows | 876.000 | info | same phrase after masking numbers (e.g. 'send 50' vs 'send 500') |
| duplicates | within_split_redundant_train | 5.000 | warn | 2155 unique of 2160 |
| duplicates | within_split_redundant_validation | 0.000 | ok | 270 unique of 270 |
| duplicates | within_split_redundant_test | 0.000 | ok | 270 unique of 270 |
| duplicates | within_split_redundant_evaluation_only | 1.000 | warn | 299 unique of 300 |

Only 2,993 of 3,000 rows are distinct after normalization (lowercase, punctuation-insensitive). Unique phrases per partition: train 2,155, validation 270, test 270, evaluation_only 299. Exact duplicates: 0 redundant rows. The remaining repeats differ only in case or punctuation. Every one of the 270 test rows is a distinct phrase.

Most repeated phrases:

| canonical | count | splits | intents |
|---|---|---|---|
| app keeps crashing asap | 2 | train | HELP_SUPPORT |
| app keeps crashing if possible | 2 | train | HELP_SUPPORT |
| app keeps crashing now | 2 | train | HELP_SUPPORT |
| app keeps crashing on sole | 2 | train | HELP_SUPPORT |
| forgot password now | 2 | train | HELP_SUPPORT |
| how tall is mount everest if possible | 2 | evaluation_only | UNKNOWN_INTENT |
| spending this month in sole | 2 | train\|validation | CHECK_SPENDING |

## Contradictory labels

0 normalized phrases carry more than one intent label — no label conflicts were found.

## Cross-partition overlap and leakage

| category | check | value | status | details |
|---|---|---|---|---|
| leakage | shared_phrases_train_vs_validation | 1.000 | warn | 1 validation rows; 18 shared number-masked templates |
| leakage | shared_phrases_train_vs_test | 0.000 | ok | 0 test rows; 36 shared number-masked templates |
| leakage | shared_phrases_train_vs_evaluation_only | 0.000 | ok | 0 evaluation_only rows; 0 shared number-masked templates |
| leakage | shared_phrases_validation_vs_test | 0.000 | ok | 0 test rows; 6 shared number-masked templates |
| leakage | shared_phrases_validation_vs_evaluation_only | 0.000 | ok | 0 evaluation_only rows; 0 shared number-masked templates |
| leakage | shared_phrases_test_vs_evaluation_only | 0.000 | ok | 0 evaluation_only rows; 0 shared number-masked templates |
| leakage | validation_rows_normalized_in_train | 1.000 | warn |  |
| leakage | validation_rows_template_in_train | 23.000 | warn |  |
| leakage | validation_rows_near_duplicate_of_train | 7.000 | info | token Jaccard >= 0.8; median max-Jaccard 0.50 |
| leakage | test_rows_normalized_in_train | 0.000 | ok |  |
| leakage | test_rows_template_in_train | 39.000 | warn |  |
| leakage | test_rows_near_duplicate_of_train | 16.000 | info | token Jaccard >= 0.8; median max-Jaccard 0.57 |
| leakage | evaluation_only_rows_normalized_in_train | 0.000 | ok |  |
| leakage | evaluation_only_rows_template_in_train | 0.000 | ok |  |
| leakage | evaluation_only_rows_near_duplicate_of_train | 0.000 | info | token Jaccard >= 0.8; median max-Jaccard 0.27 |

| Partition | Rows | Phrase seen in train | Number-masked template seen in train | Near-duplicate (Jaccard ≥ 0.8) | Leakage-flagged |
|---|---|---|---|---|---|
| validation | 270 | 1 | 23 | 7 | 23 |
| test | 270 | 0 | 39 | 16 | 39 |
| evaluation_only | 300 | 0 | 0 | 0 | 0 |

**Leakage flag:** 39 test rows (14.4%) repeat a training phrase exactly or after masking numbers. The primary benchmark keeps the supplied split; a **leakage-controlled test subset** excluding these rows is evaluated separately (`metrics_test_leakage_controlled.json`). Unknown-evaluation phrases do not overlap training.

Rows flagged per partition are listed in `data/processed_v5/leakage_rows.csv`; all shared phrases in `data/processed_v5/cross_split_overlap.csv`.

## Vocabulary shift (train vocabulary vs held-out partitions)

| split | n | rows_with_oov | share_rows_with_oov | mean_oov_token_share |
|---|---|---|---|---|
| train | 2,160 | 0 | 0.000 | 0.000 |
| validation | 270 | 29 | 0.107 | 0.032 |
| test | 270 | 31 | 0.115 | 0.032 |

11.5% of test rows contain at least one word that never occurs in training (most frequent: `look`, `owe`, `using`, `statement`, `talk`, `recent`, `customer`, `service`, `average`, `daily`, `complaint`, `kesa`). Most held-out commands are built from vocabulary seen in training.

## Implications

1. Training-set metrics and cross-validation inside the training split are optimistic (duplicates and shared templates).
2. All 270 test rows are distinct phrases; one error moves accuracy by 0.37 points.
3. Train→test overlap: 0 exact/normalized and 39 number-masked matches. Reported, not removed; see the leakage-controlled test subset.
4. Unknown-evaluation phrases repeat (299 distinct in 300 rows; e.g. `how tall is mount everest if possible` ×2); unknown rejection is also reported per unique phrase.
