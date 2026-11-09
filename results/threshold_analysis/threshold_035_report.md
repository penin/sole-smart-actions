# Threshold 0.35 — Detailed Evaluation (Baseline V1 model)

_Generated 2026-11-09T04:49:05+00:00 by `scripts/evaluate_threshold.py`. Model `models\bow_logreg_v1_tuned_v1.0.0.joblib` (SHA-256 `30d65968ec45…`), not retrained._

**Status: exploratory.** The test split and the 300 evaluation-only unknown commands were already inspected in earlier experiments (Baseline V1 evaluation and its exploratory sweep). These results describe the trade-offs of the V1 model. They are not an unbiased final evaluation, and they cannot show that any threshold is statistically optimal.

## Setup

- **Decision rule (model):** if max class probability < threshold → `UNKNOWN_INTENT`; otherwise the highest-probability supported intent.
- **Application mapping:** `UNKNOWN_INTENT` → `FALLBACK_HELP_OR_CLARIFY` (offer HELP_SUPPORT or ask the customer to rephrase); any intent → `ROUTE_<INTENT>`. A *model* classification of HELP_SUPPORT is counted separately from a rejection.
- **Data:** 270 legitimate test commands (30 per intent) and 300 unsupported evaluation-only commands.
- **Probabilities:** read from V1's saved prediction files (saved_v1_predictions) and checked against the saved model (max abs difference 0.0e+00).
- **Reproducibility check:** at the frozen V1 threshold (0.19572, reported as 0.196) this analysis reproduces V1's published outcome counts exactly. All Baseline V1 artifacts were verified unchanged (`baseline_v1_manifest.json`).

## 1. Outcomes at 0.35 vs Baseline V1

| outcome | V1 0.196 | 0.35 |
|---|---|---|
| Legitimate: correctly accepted (model routes to the right intent) | 204 | 167 |
| Legitimate: incorrectly accepted (routed to the wrong intent) | 41 | 10 |
| Legitimate: rejected (UNKNOWN_INTENT → app fallback) | 25 | 93 |
| Unsupported: correctly rejected (UNKNOWN_INTENT → app fallback) | 44 | 273 |
| Unsupported: classified HELP_SUPPORT (app routes to help) | 69 | 19 |
| Unsupported: routed to other non-financial intents | 158 | 8 |
| Unsupported: routed to TRANSFER_MONEY / BILL_PAY / WITHDRAW_MONEY | 29 | 0 |

| Rate | V1 0.196 | 0.35 |
|---|---|---|
| Known-intent coverage | 90.7% | 65.6% |
| Known-intent false rejection rate | 9.3% | 34.4% |
| Accuracy among accepted known commands | 83.3% | 94.4% |
| Macro F1 among accepted known commands | 0.825 | 0.893 |
| End-to-end known accuracy (rejections count as wrong) | 75.6% | 61.9% |
| Unknown rejection rate | 14.7% | 91.0% |
| Unknown false acceptance rate | 85.3% | 9.0% |
| Unknown rejection rate, per unique phrase | 8.5% | 82.4% |
| Safe fallback rate (rejected or HELP_SUPPORT) | 37.7% | 97.3% |
| Validation known coverage (reference) | 95.2% | 73.0% |

## 2. Effect on legitimate customer commands

- Coverage falls from 90.7% to **65.6%**: 93 of 270 legitimate commands (34.4%) now go to the fallback instead of being routed.
- Compared with V1, 68 more legitimate commands are rejected. **31** of them were previously *misrouted* (the higher threshold turns a wrong action into a fallback) and **37** were previously *correct* (a working command is now blocked).
- Commands that are accepted are more reliable: accuracy 83.3% → 94.4%, macro F1 0.825 → 0.893. End-to-end accuracy falls (75.6% → 61.9%) because rejections count as failures.
- 21 rejected commands were genuine HELP_SUPPORT requests. The fallback offers help anyway, so at application level these customers still reach support.
- The burden is uneven across intents. The most-rejected intent is **MANAGE_CONTACTS**.

| intent | rejected @ V1 0.196 | rejected @ 0.35 | support | rate @ 0.35 |
|---|---|---|---|---|
| TRANSFER_MONEY | 1 | 10 | 30 | 33.3% |
| CHECK_BALANCE | 0 | 0 | 30 | 0.0% |
| VIEW_TRANSACTIONS | 9 | 16 | 30 | 53.3% |
| WITHDRAW_MONEY | 4 | 4 | 30 | 13.3% |
| BILL_PAY | 0 | 0 | 30 | 0.0% |
| HELP_SUPPORT | 6 | 21 | 30 | 70.0% |
| MANAGE_CONTACTS | 0 | 24 | 30 | 80.0% |
| VIEW_GROUP | 4 | 5 | 30 | 16.7% |
| CHECK_SPENDING | 1 | 13 | 30 | 43.3% |

Figure: `legitimate_rejection_by_intent.png`.

Rejected legitimate commands (distinct phrases, most confident first):

| text | actual_intent | top_intent | max_confidence |
|---|---|---|---|
| Summarize my expenses now | CHECK_SPENDING | HELP_SUPPORT | 0.341 |
| Where is my statement | VIEW_TRANSACTIONS | CHECK_BALANCE | 0.332 |
| Where is my statement? | VIEW_TRANSACTIONS | CHECK_BALANCE | 0.332 |
| Help in sole | HELP_SUPPORT | HELP_SUPPORT | 0.311 |
| Edit priya's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.305 |
| Total purchases this week | CHECK_SPENDING | CHECK_SPENDING | 0.303 |
| Total purchases last week | CHECK_SPENDING | CHECK_SPENDING | 0.301 |
| Total purchases in the last 30 days | CHECK_SPENDING | CHECK_SPENDING | 0.297 |
| Edit jone's details? | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.287 |
| Edit jone's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.287 |
| Edit mika's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.287 |
| Edit roshni's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.285 |
| Edit kumar's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.285 |
| Edit tui's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.284 |
| Edit ravi's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.284 |
| Edit joeli's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.283 |
| Edit litia's details? | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.276 |
| Help at the moment | HELP_SUPPORT | HELP_SUPPORT | 0.275 |
| Edit john's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.275 |
| Edit adi's details | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.274 |

Legitimate commands still accepted but routed to the wrong intent:

| text | actual_intent | top_intent | max_confidence |
|---|---|---|---|
| Show beneficiaries | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.474 |
| Show beneficiaries? | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.474 |
| Show expenditure summary? | CHECK_SPENDING | VIEW_TRANSACTIONS | 0.458 |
| Show expenditure summary | CHECK_SPENDING | VIEW_TRANSACTIONS | 0.458 |
| Show beneficiaries quickly | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.447 |
| Where is my statement on my account | VIEW_TRANSACTIONS | CHECK_BALANCE | 0.444 |
| Show expenditure summary in sole | CHECK_SPENDING | VIEW_TRANSACTIONS | 0.387 |
| Where is my statement this time | VIEW_TRANSACTIONS | CHECK_BALANCE | 0.364 |

Most legitimate rejections are commands whose wording is absent from training (`I owe …`, `Edit <name>'s details`, `customer service`, `Recent transactions`). The model is under-confident on anything unfamiliar, whether it is in scope or not.

## 3. Effect on unsupported commands

- **273** of 300 (91.0%) are rejected and reach the fallback (V1: 44, 14.7%).
- **19** (6.3%) are *classified* as HELP_SUPPORT. That is a model error, but the application outcome is benign (the customer is sent to support).
- Safe fallback rate (rejected + HELP_SUPPORT): **97.3%** (V1: 37.7%).
- **8** are routed to other non-financial intents (e.g. balance or transaction views): wrong, but no money moves.

Unsupported commands not rejected:

| text | top_intent | max_confidence | outcome |
|---|---|---|---|
| Turn on the lights on sole | CHECK_SPENDING | 0.580 | accepted_as_other_intent |
| Renew my passport now now | HELP_SUPPORT | 0.520 | accepted_as_help_support |
| How tall is mount everest on my account | CHECK_BALANCE | 0.493 | accepted_as_other_intent |
| Predict the lottery please on my account | HELP_SUPPORT | 0.462 | accepted_as_help_support |
| Renew my passport if possible | HELP_SUPPORT | 0.451 | accepted_as_help_support |
| Find a cinema on the app | VIEW_TRANSACTIONS | 0.431 | accepted_as_other_intent |
| Find a hotel on the app | VIEW_TRANSACTIONS | 0.431 | accepted_as_other_intent |
| Find a curry recipe on the app | VIEW_TRANSACTIONS | 0.431 | accepted_as_other_intent |
| Renew my passport on the app | HELP_SUPPORT | 0.430 | accepted_as_help_support |
| What's on tv this time | CHECK_SPENDING | 0.427 | accepted_as_other_intent |
| Renew my passport on sole | HELP_SUPPORT | 0.421 | accepted_as_help_support |
| Order food please if possible | HELP_SUPPORT | 0.414 | accepted_as_help_support |
| Renew my passport please | HELP_SUPPORT | 0.410 | accepted_as_help_support |
| Renew my passport now | HELP_SUPPORT | 0.404 | accepted_as_help_support |
| Write my homework now | HELP_SUPPORT | 0.404 | accepted_as_help_support |
| Book a taxi right now on my account | HELP_SUPPORT | 0.398 | accepted_as_help_support |
| Turn on the lights this time | CHECK_SPENDING | 0.388 | accepted_as_other_intent |
| Who won the rugby please if possible | HELP_SUPPORT | 0.383 | accepted_as_help_support |
| Order food on my account | HELP_SUPPORT | 0.382 | accepted_as_help_support |
| What's the exchange rate please on my account | HELP_SUPPORT | 0.381 | accepted_as_help_support |
| Order food please quickly | HELP_SUPPORT | 0.376 | accepted_as_help_support |
| What time is the game in sole | CHECK_BALANCE | 0.373 | accepted_as_other_intent |
| Order food please now | HELP_SUPPORT | 0.366 | accepted_as_help_support |
| Who won the rugby please quickly | HELP_SUPPORT | 0.353 | accepted_as_help_support |
| Order food please on sole | HELP_SUPPORT | 0.352 | accepted_as_help_support |
| Who won the rugby please on sole | HELP_SUPPORT | 0.350 | accepted_as_help_support |
| Order food please asap | HELP_SUPPORT | 0.350 | accepted_as_help_support |

## 4. Financial-action routing

- Unsupported commands routed to TRANSFER_MONEY / BILL_PAY / WITHDRAW_MONEY: **0** at 0.35 vs 29 at V1 (TRANSFER_MONEY 14, BILL_PAY 15, WITHDRAW_MONEY 0).
- Legitimate commands misrouted *into* a different money-moving intent: 0 at 0.35 vs 8 at V1.
- A threshold reduces this risk but cannot remove it for new phrasings. Money-moving intents should still require explicit confirmation of amount and payee/biller.

## 5. Is 0.35 a reasonable candidate?

**It is a defensible candidate for further testing. This analysis cannot show it is optimal.** On these (previously inspected) sets, 0.35 sits where unknown rejection has largely saturated (91.0%) and unknown→financial routing has reached 0. The price is high: about one in 3 legitimate commands goes to the fallback, and certain intents are disproportionately blocked.

Considerations:

- **Selection evidence is weak.** The choice is informed by the same test and unknown sets reported here. On validation, 0.35 accepts only 73.0% of known commands, far from V1's prespecified 95% coverage target. The original protocol would not have selected it.
- **The model is under-confident.** Baseline V1 test ECE is 0.30, mean top-label confidence 0.49 vs accuracy 0.79. So 0.35 here does not mean 35% confidence in a calibrated sense, and the value will not transfer to the CNN/transformer models.
- **The unknown set is synthetic and repetitive.** Phrases like `Order food please …` and `Renew my passport …` repeat with filler suffixes, so the 300 rows behave like far fewer independent tests.
- **Acceptability depends on the fallback UX.** If the fallback is a quick *"Did you mean …?"* prompt showing the top intent(s), a ~34% fallback rate is irritating but recoverable. If the fallback dead-ends the customer, it is likely unacceptable.
- **Better levers exist** than the threshold alone: calibrate probabilities, add sub-word/semantic features so legitimate paraphrases gain confidence, and add a dedicated out-of-scope detector.

## How a production threshold should be selected

1. **Freeze the model** and the decision rule (`max_probability < t → UNKNOWN_INTENT`).
2. **Build a calibration set** that has not been used before: legitimate commands for every intent plus a *separate* pool of unsupported commands, ideally real or realistic Fijian customer phrasing. Split unknowns by phrase/template so no phrase appears in two sets.
3. **Prespecify the selection rule** before looking at results, e.g. *highest coverage subject to ≥ 90% unknown rejection and zero unknowns routed to money-moving intents*, or a cost-weighted objective where routing an unsupported request to TRANSFER_MONEY/BILL_PAY/WITHDRAW_MONEY costs far more than a fallback.
4. **Optionally calibrate probabilities first** (temperature scaling on the calibration set), so the threshold has a stable meaning.
5. **Choose t on the calibration set only**, then **evaluate once on a new, untouched test set** (legitimate + unsupported) and report coverage, rejection and financial-routing rates with phrase-clustered confidence intervals.
6. **Monitor after launch.** Track fallback rate and user rephrasing, and re-run steps 2–5 when the model or customer language changes.

## All thresholds (exploratory)

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

Figures: `threshold_comparison.png`, `coverage_vs_rejection.png`, `safe_fallback_comparison.png`, `financial_routing_risk.png`, `legitimate_rejection_by_intent.png`.

## Files

- `threshold_035_predictions.csv`: every legitimate test and unsupported command with probabilities, model decision, application action and outcome.
- `threshold_035_errors.csv`: all non-correct outcomes (incorrect accepts, legitimate rejections, unsupported commands not rejected).
- `threshold_035_vs_v1_changes.csv`: rows whose outcome differs from Baseline V1.
