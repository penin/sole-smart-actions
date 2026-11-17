# V4 vs V5: Bag of Words + Logistic Regression

_Generated 2026-11-12T02:55:57+00:00. V4: `v4_cleaned` (outputs in `results/`, read-only). V5: `v5_deduplicated` (outputs in `results_v5/`). Same code, hyperparameter grid, preprocessing, seed (42) and selection rules; only the dataset differs._

## Headline comparison

Raw test accuracy counts every one of the legitimate test commands, with no confidence threshold applied. Threshold rows use each version's own validation-calibrated default (95% validation coverage target).

| metric | V4 | V5 | change |
|---|---|---|---|
| Training accuracy | 98.7% | 100.0% | +1.3 pts |
| Validation accuracy | 75.6% | 91.9% | +16.3 pts |
| Raw test accuracy (no threshold) | 79.3% | 93.7% | +14.4 pts |
| Test macro F1 | 0.796 | 0.937 | +0.140 |
| Test weighted F1 | 0.796 | 0.937 | +0.140 |
| Test macro ROC-AUC (OvR) | 0.954 | 0.995 | +0.041 |
| Validation-calibrated threshold | 19.6% | 34.2% | +14.6 pts |
| Known-intent coverage at default (test) | 90.7% | 96.3% | +5.6 pts |
| Unknown rejection rate at default | 14.7% | 29.3% | +14.7 pts |
| Unknown safe-fallback rate at default | 37.7% | 56.0% | +18.3 pts |
| Inference latency P50 (single request) | 0.144 ms | 0.280 ms | +0.136 ms |
| Inference latency P95 (single request) | 0.173 ms | 0.444 ms | +0.272 ms |
| Model size | 23.3 KB | 222.1 KB | +198.8 KB |

Additional detail:

| metric | V4 | V5 | change |
|---|---|---|---|
| Correct / total test commands | 214 / 270 | 253 / 270 | +39 |
| Test macro F1, 95% phrase-clustered CI | 0.712–0.864 | 0.905–0.964 |  |
| Leakage-controlled test macro F1 | 0.774 (n=249) | 0.926 (n=231) |  |
| Unknowns routed to money-moving intents at default | 29 | 23 | -6 |
| Selected hyperparameters | ngram(1,1), C=0.1, counts | ngram(1,2), C=10, binary |  |
| Vocabulary size | 250 | 2,584 | +2334 |
| Training time (final fit) | 39 ms | 175 ms |  |

## Controlled comparison: same test commands, both models

Because the test sets differ, both trained models were also scored on identical commands taken from the V5 test set. The V4 model was loaded read-only from `models/bow_logreg_v1_tuned_v1.0.0.joblib`.

| test subset | n | V4 model | V5 model |
|---|---|---|---|
| Commands in BOTH test sets (identical examples) | 173 | 82.7% | 90.2% |
| V5 replacement commands (new in V5) | 97 | 76.3% | 100.0% |
|   …of which match a training command once numbers + months are masked | 61 | 78.7% | 100.0% |
| All V5 test commands | 270 | 80.4% | 93.7% |

- **Like-for-like improvement:** on the 173 commands present in both test sets, accuracy goes from 82.7% (V4 model) to 90.2% (V5 model), +7.5 points. This is the change attributable to the new training data rather than to a different test set.
- **The replacement commands are easier:** the V5 model scores 100.0% on the 97 replacement test commands. 61 of them match a training command once numbers and month names are masked (e.g. test "Display my wallet balance on 15 September" vs training "Display my wallet balance on 7 November"). The deduplication removed exact repeats but introduced template-generated commands shared across splits. That is a milder form of the same leakage, and it inflates the headline test accuracy.
- **Headline vs like-for-like:** raw test accuracy rose 14.4 points, but only 7.5 of that is visible on identical examples. Report V5 accuracy on the 173 original commands (90.2%) alongside the headline figure.
- Per-command detail: `results_v5/metrics/test_original_vs_replacement_commands.csv`.

## Read this before comparing the numbers

**V4 and V5 are evaluated on different test examples.** Only 173 of the 270 V5 test commands also appear in the V4 test set (after normalization); the rest were replaced when duplicates were removed. A change in test accuracy therefore mixes three effects: (1) the model learning from different training data, (2) a different, possibly harder or easier, test set, and (3) the removal of train→test duplicates that inflated V4. It cannot be attributed to model improvement alone. With 270 test commands, one command is worth 0.37 accuracy points. See the controlled comparison above.

- V4 test set: 179 distinct phrases among 270 rows; 21 rows repeated a training phrase (exact or number-masked).
- V5 test set: 270 distinct phrases among 270 rows; 0 rows repeat a training phrase exactly, 39 once numbers are masked.
- The macro F1 95% confidence intervals do not overlap (V4 0.712–0.864, V5 0.905–0.964).

## Is there overfitting?

| | Train acc. | Validation acc. | Test acc. | Train − test |
|---|---|---|---|---|
| V4 | 98.7% | 75.6% | 79.3% | 19.4 pts |
| V5 | 100.0% | 91.9% | 93.7% | 6.3 pts |

The train–test gap narrowed from 19.4 to 6.3 points. Training accuracy near 100% is expected for a linear model on 2,160 short commands with 2,584 features (V4: 250), so the gap measures how well the training vocabulary transfers to new phrasings more than classic memorization. Inside the V5 training split, grouped 5-fold cross-validation reaches 0.994 macro F1 at 1728 examples, versus 0.917 on validation and 0.937 on test. The held-out splits are clearly harder than the training distribution, so the remaining gap reflects distribution shift between splits (new paraphrase templates) as well as fitting to training phrases.

## Did duplicate removal affect generalization?

- V4 contained 1004 redundant rows under a punctuation-insensitive key; V5 contains 7 (pairs differing only by interior punctuation, e.g. "App keeps crashing? asap" / "App keeps crashing asap"; 0 under the required surrounding-punctuation definition). Removing duplicates means each training example is a distinct phrasing, and test scores can no longer be inflated by memorized repeats.
- Raw test accuracy rose by 14.4 points (214/270 → 253/270). Because the test sets differ (see above), this is evidence about the *combined* effect of the new data, not a controlled measurement of deduplication.
- The leakage-controlled test subsets (rows not matching any training phrase) give macro F1 0.774 (V4) vs 0.926 (V5); this is the fairest single comparison of generalization to unseen commands.

## Caveats

- Both datasets are synthetic; neither test set measures real Fijian customer language.
- Each version's threshold was calibrated on its own validation split. The unknown-command set is different in distinct-phrase count (153 vs 299), so unknown-rejection rates are also not strictly comparable.
- Latency is a single-machine wall-clock measurement and varies run to run.
