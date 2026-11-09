# Confidence Threshold Comparison — Baseline V1 model

_Generated 2026-11-09T04:49:05+00:00 by `scripts/evaluate_threshold.py`. No retraining; Baseline V1 artifacts verified unchanged._

**Status: exploratory.** The test split and the 300 evaluation-only unknown commands were already inspected in earlier experiments (Baseline V1 evaluation and its exploratory sweep). These results describe the trade-offs of the V1 model. They are not an unbiased final evaluation, and they cannot show that any threshold is statistically optimal.

## Comparison table

Legitimate commands: 270 test rows. Unsupported: 300 evaluation-only rows. *Model classification* metrics (coverage, accuracy, rejection) are separate from *application* metrics (safe fallback = rejected or classified HELP_SUPPORT).

| threshold | known coverage | known false-reject | accepted acc | accepted macro F1 | end-to-end known acc | unknown rejected | unknown false-accept | unknown→HELP (n) | unknown→HELP (%) | safe fallback | unknown→financial (n) | val coverage (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.196 (V1 frozen) | 90.7% | 9.3% | 83.3% | 82.5% | 75.6% | 14.7% | 85.3% | 69 | 23.0% | 37.7% | 29 | 95.2% |
| 0.25 | 82.2% | 17.8% | 88.3% | 87.5% | 72.6% | 38.0% | 62.0% | 58 | 19.3% | 57.3% | 26 | 90.7% |
| 0.30 | 68.9% | 31.1% | 91.9% | 88.3% | 63.3% | 70.3% | 29.7% | 24 | 8.0% | 78.3% | 18 | 85.2% |
| 0.35 [candidate] | 65.6% | 34.4% | 94.4% | 89.3% | 61.9% | 91.0% | 9.0% | 19 | 6.3% | 97.3% | 0 | 73.0% |
| 0.40 | 61.1% | 38.9% | 95.2% | 90.1% | 58.1% | 95.0% | 5.0% | 9 | 3.0% | 98.0% | 0 | 68.1% |
| 0.45 | 55.2% | 44.8% | 96.0% | 83.6% | 53.0% | 98.3% | 1.7% | 3 | 1.0% | 99.3% | 0 | 59.6% |
| 0.50 | 51.5% | 48.5% | 100.0% | 100.0% | 51.5% | 99.3% | 0.7% | 1 | 0.3% | 99.7% | 0 | 51.9% |
| 0.60 | 41.1% | 58.9% | 100.0% | 100.0% | 41.1% | 100.0% | 0.0% | 0 | 0.0% | 100.0% | 0 | 38.9% |

## Unsupported commands routed to money-moving intents

| label | unknown_to_TRANSFER_MONEY | unknown_to_BILL_PAY | unknown_to_WITHDRAW_MONEY | unknown_to_financial | known_misrouted_to_financial |
|---|---|---|---|---|---|
| 0.196 (V1 frozen) | 14 | 15 | 0 | 29 | 8 |
| 0.25 | 14 | 12 | 0 | 26 | 5 |
| 0.30 | 8 | 10 | 0 | 18 | 0 |
| 0.35 [candidate] | 0 | 0 | 0 | 0 | 0 |
| 0.40 | 0 | 0 | 0 | 0 | 0 |
| 0.45 | 0 | 0 | 0 | 0 | 0 |
| 0.50 | 0 | 0 | 0 | 0 | 0 |
| 0.60 | 0 | 0 | 0 | 0 | 0 |

## Legitimate commands rejected, by actual intent (count of 30)

| intent | V1 0.196 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|
| TRANSFER_MONEY | 1 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| CHECK_BALANCE | 0 | 0 | 0 | 0 | 5 | 10 | 11 | 27 |
| VIEW_TRANSACTIONS | 9 | 11 | 12 | 16 | 17 | 18 | 18 | 19 |
| WITHDRAW_MONEY | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 8 |
| BILL_PAY | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| HELP_SUPPORT | 6 | 10 | 20 | 21 | 24 | 24 | 24 | 24 |
| MANAGE_CONTACTS | 0 | 1 | 23 | 24 | 24 | 27 | 30 | 30 |
| VIEW_GROUP | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| CHECK_SPENDING | 1 | 7 | 10 | 13 | 16 | 23 | 29 | 30 |

## How a production threshold should be selected

1. **Freeze the model** and the decision rule (`max_probability < t → UNKNOWN_INTENT`).
2. **Build a calibration set** that has not been used before: legitimate commands for every intent plus a *separate* pool of unsupported commands, ideally real or realistic Fijian customer phrasing. Split unknowns by phrase/template so no phrase appears in two sets.
3. **Prespecify the selection rule** before looking at results, e.g. *highest coverage subject to ≥ 90% unknown rejection and zero unknowns routed to money-moving intents*, or a cost-weighted objective where routing an unsupported request to TRANSFER_MONEY/BILL_PAY/WITHDRAW_MONEY costs far more than a fallback.
4. **Optionally calibrate probabilities first** (temperature scaling on the calibration set), so the threshold has a stable meaning.
5. **Choose t on the calibration set only**, then **evaluate once on a new, untouched test set** (legitimate + unsupported) and report coverage, rejection and financial-routing rates with phrase-clustered confidence intervals.
6. **Monitor after launch.** Track fallback rate and user rephrasing, and re-run steps 2–5 when the model or customer language changes.

Full per-threshold details: `threshold_comparison.json` / `.csv`; dense 0.01-step sweep: `dense_threshold_sweep.csv`; candidate detail: `threshold_035_report.md`.
