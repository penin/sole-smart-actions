# Error Analysis — BoW + Logistic Regression (selected model)

_Generated 2026-11-09T04:20:40+00:00. Threshold for rejection: 0.1957. Row-level predictions with all class probabilities: `results/predictions/*_predictions.csv`. Tables below are also saved as `results/metrics/error_*.csv`._

## 1. Most common confusions

| split | actual_intent | predicted_intent | count | unique_phrases | example |
|---|---|---|---|---|---|
| test | TRANSFER_MONEY | HELP_SUPPORT | 8 | 8 | I owe joeli 100 |
| test | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 7 | 5 | View my recipient list? |
| test | CHECK_SPENDING | VIEW_TRANSACTIONS | 6 | 4 | Show expenditure summary |
| test | HELP_SUPPORT | BILL_PAY | 6 | 3 | I have a complaint |
| test | HELP_SUPPORT | MANAGE_CONTACTS | 6 | 2 | Need customer service |
| test | VIEW_TRANSACTIONS | CHECK_BALANCE | 6 | 4 | Where is my statement |
| test | VIEW_GROUP | VIEW_TRANSACTIONS | 4 | 1 | View groupwealth |
| test | WITHDRAW_MONEY | MANAGE_CONTACTS | 4 | 2 | Prepare a withdrawal |
| test | CHECK_SPENDING | HELP_SUPPORT | 3 | 2 | Summarize my expenses |
| test | VIEW_TRANSACTIONS | HELP_SUPPORT | 3 | 3 | Recent transactions now |
| test | TRANSFER_MONEY | WITHDRAW_MONEY | 2 | 2 | I owe tui $200 |
| test | VIEW_GROUP | HELP_SUPPORT | 1 | 1 | View groupwealth now |
| validation | CHECK_BALANCE | HELP_SUPPORT | 18 | 15 | Do i have 5 |
| validation | HELP_SUPPORT | CHECK_BALANCE | 12 | 9 | My account is locked |
| validation | HELP_SUPPORT | TRANSFER_MONEY | 12 | 6 | Talk to an agent |

Train confusions (in-sample):

| actual_intent | predicted_intent | count | example |
|---|---|---|---|
| TRANSFER_MONEY | BILL_PAY | 15 | Pay joeli 80 |
| CHECK_BALANCE | CHECK_SPENDING | 5 | How much can i spend |
| VIEW_GROUP | VIEW_TRANSACTIONS | 5 | Show my groups |
| BILL_PAY | HELP_SUPPORT | 1 | Bill payment if possible |
| CHECK_BALANCE | VIEW_TRANSACTIONS | 1 | Show balance at the moment |
| VIEW_GROUP | CHECK_BALANCE | 1 | Show my groups in sole |

## 2. Confident errors

Errors with confidence ≥ 0.9: 0 on validation/test. The selected model's highest held-out confidence is 0.959, so the more useful view is **errors that the rejection threshold would accept** (distinct phrases, highest confidence first):

| split | text | actual_intent | predicted_intent | max_confidence | top2_margin | oov_tokens |
|---|---|---|---|---|---|---|
| validation | My account is locked in sole | HELP_SUPPORT | CHECK_BALANCE | 0.575 | 0.362 | locked |
| validation | Find my transfer to ana | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.574 | 0.437 |  |
| validation | My money is missing on my account | HELP_SUPPORT | CHECK_BALANCE | 0.563 | 0.334 | missing |
| validation | How much did i contribute? | VIEW_GROUP | CHECK_SPENDING | 0.552 | 0.401 | contribute |
| validation | How much did i contribute | VIEW_GROUP | CHECK_SPENDING | 0.552 | 0.401 | contribute |
| validation | Find my transfer to mary | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.536 | 0.386 |  |
| validation | Find my transfer to mere | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.528 | 0.377 |  |
| validation | Do i have 250 on my account | CHECK_BALANCE | HELP_SUPPORT | 0.525 | 0.380 |  |
| validation | Find my transfer to peni | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.522 | 0.365 |  |
| validation | Find my transfer to sera | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.514 | 0.353 |  |
| validation | Find my transfer to adi | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.511 | 0.356 |  |
| validation | Find my transfer to asha | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.497 | 0.334 |  |
| validation | Find my transfer to roshni | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.496 | 0.336 |  |
| validation | Find my transfer to ravi | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.483 | 0.311 |  |
| validation | Find my transfer to raj | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.477 | 0.309 |  |
| validation | Find my transfer to ben | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.476 | 0.305 |  |
| test | Show beneficiaries | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.474 | 0.362 | beneficiaries |
| test | Show beneficiaries? | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.474 | 0.362 | beneficiaries |
| test | Show expenditure summary? | CHECK_SPENDING | VIEW_TRANSACTIONS | 0.458 | 0.346 | expenditure |
| test | Show expenditure summary | CHECK_SPENDING | VIEW_TRANSACTIONS | 0.458 | 0.346 | expenditure |

## 3. Low-confidence correct predictions (would be falsely rejected)

11 distinct correct validation/test phrases fall below the threshold:

| split | text | actual_intent | predicted_intent | max_confidence | top2_margin | oov_tokens |
|---|---|---|---|---|---|---|
| validation | Need assistance with login this time | HELP_SUPPORT | HELP_SUPPORT | 0.156 | 0.017 | assistance with login |
| validation | Look up ana | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.177 | 0.007 | look |
| test | Total purchases | CHECK_SPENDING | CHECK_SPENDING | 0.177 | 0.019 | purchases |
| validation | Look up sera | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.179 | 0.007 | look |
| test | Recent transactions? | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.181 | 0.011 | recent |
| test | Recent transactions | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.181 | 0.011 | recent |
| validation | Look up savenaca | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.184 | 0.012 | look |
| validation | Look up mere | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.184 | 0.013 | look |
| validation | Look up adi | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.185 | 0.014 | look |
| validation | Look up ravi | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.191 | 0.022 | look |
| validation | Look up tui | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.195 | 0.028 | look |

## 4. Ambiguous commands (top-2 margin < 0.1)

| split | text | actual_intent | predicted_intent | second_intent | top2_margin | correct |
|---|---|---|---|---|---|---|
| test | Recent transactions on the app | VIEW_TRANSACTIONS | HELP_SUPPORT | VIEW_TRANSACTIONS | 0.003 | no |
| validation | Need assistance with login | HELP_SUPPORT | MANAGE_CONTACTS | HELP_SUPPORT | 0.005 | no |
| test | Need customer service | HELP_SUPPORT | MANAGE_CONTACTS | HELP_SUPPORT | 0.005 | no |
| validation | Need assistance with login? | HELP_SUPPORT | MANAGE_CONTACTS | HELP_SUPPORT | 0.005 | no |
| test | Need customer service? | HELP_SUPPORT | MANAGE_CONTACTS | HELP_SUPPORT | 0.005 | no |
| validation | Look up ana | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.007 | yes |
| validation | Look up sera | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.007 | yes |
| validation | Talk to an agent on sole | HELP_SUPPORT | TRANSFER_MONEY | CHECK_SPENDING | 0.008 | no |
| test | Recent transactions on sole | VIEW_TRANSACTIONS | HELP_SUPPORT | CHECK_SPENDING | 0.011 | no |
| test | Recent transactions | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | MANAGE_CONTACTS | 0.011 | yes |
| test | Recent transactions? | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | MANAGE_CONTACTS | 0.011 | yes |
| validation | Look up savenaca | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.012 | yes |
| validation | Look up mere | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.013 | yes |
| validation | Look up adi | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.014 | yes |
| test | I owe mere 50 | TRANSFER_MONEY | HELP_SUPPORT | WITHDRAW_MONEY | 0.014 | no |
| test | Total purchases in the last 30 days | CHECK_SPENDING | CHECK_SPENDING | VIEW_TRANSACTIONS | 0.015 | yes |
| validation | I need to pay broadband | BILL_PAY | BILL_PAY | TRANSFER_MONEY | 0.015 | yes |
| test | I owe joeli 100 | TRANSFER_MONEY | HELP_SUPPORT | WITHDRAW_MONEY | 0.015 | no |
| validation | I need to pay efl | BILL_PAY | BILL_PAY | TRANSFER_MONEY | 0.015 | yes |
| validation | Do i have $20 | CHECK_BALANCE | HELP_SUPPORT | WITHDRAW_MONEY | 0.016 | no |

## 5. Short commands (≤ 2 tokens) and out-of-vocabulary terms

| split | n | short_n | short_accuracy | longer_accuracy | rows_with_oov | share_rows_with_oov | oov_accuracy | in_vocab_accuracy | mean_oov_token_share |
|---|---|---|---|---|---|---|---|---|---|
| train | 2,160 | 178 | 1.000 | 0.986 | 0 | 0.000 | n/a | 0.987 | 0.000 |
| validation | 270 | 15 | 0.867 | 0.749 | 88 | 0.326 | 0.591 | 0.835 | 0.105 |
| test | 270 | 44 | 0.841 | 0.783 | 125 | 0.463 | 0.616 | 0.945 | 0.136 |

Short-command errors (validation/test):

| split | text | actual_intent | predicted_intent | max_confidence | top2_margin | oov_tokens |
|---|---|---|---|---|---|---|
| validation | Open statement | VIEW_TRANSACTIONS | VIEW_GROUP | 0.186 | 0.027 | statement |
| test | View groupwealth | VIEW_GROUP | VIEW_TRANSACTIONS | 0.181 | 0.022 |  |
| test | Show beneficiaries | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.474 | 0.362 | beneficiaries |
| test | Show beneficiaries? | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.474 | 0.362 | beneficiaries |

Most frequent out-of-vocabulary tokens in validation/test (never seen in training):

| oov_token | held_out_occurrences |
|---|---|
| edit | 21 |
| agent | 16 |
| has | 13 |
| received | 12 |
| talk | 12 |
| with | 12 |
| recent | 12 |
| customer | 12 |
| service | 12 |
| purchases | 12 |
| look | 11 |
| available | 11 |
| move | 10 |
| owe | 10 |
| statement | 9 |
| using | 8 |
| locked | 6 |
| missing | 6 |
| assistance | 6 |
| login | 6 |
| complaint | 6 |
| average | 4 |
| daily | 4 |
| prepare | 4 |
| expenditure | 4 |

Out-of-vocabulary words explain most errors. Rows with any unseen word are far less accurate than fully in-vocabulary rows. The vectorizer silently drops unseen words, so `I owe jone 80` reduces to `i jone 80`, which carries almost no intent signal.

## 6. Word-order blindness

| text_a | pred_a | conf_a | text_b | pred_b | conf_b | same_bag_of_words | max_abs_prob_diff |
|---|---|---|---|---|---|---|---|
| send money to jone | TRANSFER_MONEY | 0.814 | jone send money to | TRANSFER_MONEY | 0.814 | yes | 0.000 |
| jone sent me money | VIEW_TRANSACTIONS | 0.322 | i sent jone money | HELP_SUPPORT | 0.175 | no | 0.188 |
| pay bill then check balance | BILL_PAY | 0.736 | check balance then pay bill | BILL_PAY | 0.736 | yes | 0.000 |
| transfer from mere to jone | TRANSFER_MONEY | 0.771 | transfer from jone to mere | TRANSFER_MONEY | 0.771 | yes | 0.000 |
| did i send money to mere | TRANSFER_MONEY | 0.728 | i did send money to mere | TRANSFER_MONEY | 0.728 | yes | 0.000 |

Pairs with the same bag of words receive **identical** probabilities (max |Δp| = 0). The model cannot tell who pays whom (`transfer from mere to jone` vs `from jone to mere`) or the order of a multi-step request. Payer/payee direction must come from a slot-filling step, not from this classifier.

## 7. Train/test leakage

| split | leakage_flag | n | accuracy |
|---|---|---|---|
| validation | no | 267 | 0.753 |
| validation | yes | 3 | 1.000 |
| test | no | 249 | 0.775 |
| test | yes | 21 | 1.000 |

Test rows that repeat a training phrase (exact or number-masked):

| text | actual_intent | predicted_intent | max_confidence | correct |
|---|---|---|---|---|
| Show last 5 transactions | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.739 | yes |
| Pay telecom | BILL_PAY | BILL_PAY | 0.732 | yes |
| Pay waf | BILL_PAY | BILL_PAY | 0.742 | yes |
| Pay waf? | BILL_PAY | BILL_PAY | 0.742 | yes |
| Pay efl | BILL_PAY | BILL_PAY | 0.722 | yes |
| Show transactions | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.613 | yes |
| Pay water | BILL_PAY | BILL_PAY | 0.736 | yes |
| Show last 5 transactions on my account | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.593 | yes |
| Pay internet | BILL_PAY | BILL_PAY | 0.735 | yes |
| Pay school fees | BILL_PAY | BILL_PAY | 0.810 | yes |
| Pay broadband | BILL_PAY | BILL_PAY | 0.720 | yes |
| Pay phone | BILL_PAY | BILL_PAY | 0.716 | yes |
| Show transactions? | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.613 | yes |
| Pay mobile | BILL_PAY | BILL_PAY | 0.701 | yes |
| Pay electricity | BILL_PAY | BILL_PAY | 0.749 | yes |
| Show last 5 transactions? | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.739 | yes |

## 8. Financial-safety view

TRANSFER_MONEY ↔ BILL_PAY confusions:

| split | actual | predicted | count | accepted_at_threshold |
|---|---|---|---|---|
| train | TRANSFER_MONEY | BILL_PAY | 15 | 15 |
| train | BILL_PAY | TRANSFER_MONEY | 0 | 0 |
| validation | TRANSFER_MONEY | BILL_PAY | 0 | 0 |
| validation | BILL_PAY | TRANSFER_MONEY | 0 | 0 |
| test | TRANSFER_MONEY | BILL_PAY | 0 | 0 |
| test | BILL_PAY | TRANSFER_MONEY | 0 | 0 |

Training TRANSFER_MONEY rows predicted BILL_PAY (in-sample):

| text | max_confidence | final_intent |
|---|---|---|
| Pay joeli 80 | 0.460 | BILL_PAY |
| Pay ana 10? | 0.455 | BILL_PAY |
| Pay litia 60 | 0.448 | BILL_PAY |
| Pay mary 15 | 0.494 | BILL_PAY |
| Pay litia 10 | 0.448 | BILL_PAY |
| Pay mika 250 | 0.525 | BILL_PAY |
| Pay tui $100 | 0.463 | BILL_PAY |
| Pay ana 5 | 0.476 | BILL_PAY |
| Pay ana 50 | 0.449 | BILL_PAY |
| Pay mary 150 | 0.505 | BILL_PAY |
| Pay tui 60 | 0.478 | BILL_PAY |
| Pay adi $200 | 0.438 | BILL_PAY |
| Pay mere 30 | 0.495 | BILL_PAY |
| Pay litia 150 | 0.474 | BILL_PAY |
| Pay peni 80 | 0.499 | BILL_PAY |

These are person-to-person payments phrased as `pay <name> <amount>`. Because `pay` is the strongest BILL_PAY feature (see `13_top_features.png`) and names are sparse, the model routes them to bill payment. A transfer to a person mistaken for a bill payment (or vice versa) is the most consequential confusion for a wallet. A payee/biller slot check should gate both intents.

Supported requests wrongly routed **and accepted** into a money-moving intent (validation/test):

| split | text | actual_intent | predicted_intent | max_confidence | top2_margin | oov_tokens |
|---|---|---|---|---|---|---|
| validation | Find my transfer to adi | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.511 | 0.356 |  |
| validation | Talk to an agent | HELP_SUPPORT | TRANSFER_MONEY | 0.330 | 0.204 | talk agent |
| validation | Find my transfer to raj | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.477 | 0.309 |  |
| validation | Find my transfer to asha | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.497 | 0.334 |  |
| validation | Talk to an agent? | HELP_SUPPORT | TRANSFER_MONEY | 0.330 | 0.204 | talk agent |
| validation | Find my transfer to ben | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.476 | 0.305 |  |
| validation | Find my transfer to peni | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.522 | 0.365 |  |
| validation | Find my transfer to roshni | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.496 | 0.336 |  |
| validation | Talk to an agent on sole | HELP_SUPPORT | TRANSFER_MONEY | 0.237 | 0.008 | talk agent |
| validation | Talk to an agent now | HELP_SUPPORT | TRANSFER_MONEY | 0.305 | 0.180 | talk agent |
| validation | Talk to an agent please | HELP_SUPPORT | TRANSFER_MONEY | 0.303 | 0.173 | talk agent |
| validation | Talk to an agent if possible | HELP_SUPPORT | TRANSFER_MONEY | 0.294 | 0.145 | talk agent |
| validation | Find my transfer to mary | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.536 | 0.386 |  |
| validation | Find my transfer to ravi | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.483 | 0.311 |  |
| validation | Find my transfer to ana | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.574 | 0.437 |  |
| validation | Find my transfer to sera | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.514 | 0.353 |  |
| validation | Find my transfer to mere | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.528 | 0.377 |  |
| test | I have a complaint | HELP_SUPPORT | BILL_PAY | 0.263 | 0.141 | complaint |
| test | I have a complaint? | HELP_SUPPORT | BILL_PAY | 0.263 | 0.141 | complaint |
| test | I owe tui $200 | TRANSFER_MONEY | WITHDRAW_MONEY | 0.246 | 0.056 | owe |
| test | I have a complaint at the moment | HELP_SUPPORT | BILL_PAY | 0.221 | 0.073 | complaint |
| test | I owe mary $50 | TRANSFER_MONEY | WITHDRAW_MONEY | 0.248 | 0.049 | owe |

Unsupported (UNKNOWN_INTENT) requests accepted as money-moving intents (distinct phrases, 18):

| text | final_intent | max_confidence |
|---|---|---|
| Buy me a phone? | BILL_PAY | 0.316 |
| Buy me a phone | BILL_PAY | 0.316 |
| Translate this to french please in sole | TRANSFER_MONEY | 0.310 |
| Buy me a phone if possible | BILL_PAY | 0.306 |
| Buy me a phone now | BILL_PAY | 0.306 |
| Translate this to french | TRANSFER_MONEY | 0.304 |
| Translate this to french? | TRANSFER_MONEY | 0.304 |
| Buy me a phone asap | BILL_PAY | 0.302 |
| Buy me a phone please | BILL_PAY | 0.297 |
| Buy me a phone quickly | BILL_PAY | 0.296 |
| Translate this to french asap | TRANSFER_MONEY | 0.294 |
| Translate this to french quickly | TRANSFER_MONEY | 0.289 |
| Translate this to french now | TRANSFER_MONEY | 0.281 |
| Translate this to french if possible | TRANSFER_MONEY | 0.271 |
| Translate this to french this time | TRANSFER_MONEY | 0.261 |
| Buy me a phone in sole | BILL_PAY | 0.233 |
| Buy me a phone on sole | BILL_PAY | 0.229 |
| Tell me a joke if possible | BILL_PAY | 0.197 |

Most confident accepted unknowns (all intents):

| text | final_intent | max_confidence |
|---|---|---|
| Turn on the lights on sole | CHECK_SPENDING | 0.580 |
| Renew my passport now now | HELP_SUPPORT | 0.520 |
| How tall is mount everest on my account | CHECK_BALANCE | 0.493 |
| Predict the lottery please on my account | HELP_SUPPORT | 0.462 |
| Renew my passport if possible | HELP_SUPPORT | 0.451 |
| Find a curry recipe on the app | VIEW_TRANSACTIONS | 0.431 |
| Find a cinema on the app | VIEW_TRANSACTIONS | 0.431 |
| Find a hotel on the app | VIEW_TRANSACTIONS | 0.431 |
| Renew my passport on the app | HELP_SUPPORT | 0.430 |
| What's on tv this time | CHECK_SPENDING | 0.427 |
| Renew my passport on sole | HELP_SUPPORT | 0.421 |
| Order food please if possible | HELP_SUPPORT | 0.414 |
| Renew my passport please | HELP_SUPPORT | 0.410 |
| Write my homework now | HELP_SUPPORT | 0.404 |
| Renew my passport now | HELP_SUPPORT | 0.404 |

Any production use would need an explicit confirmation step (amount, payee, biller) before a money-moving action. Classifier confidence alone is insufficient.

## 9. Qualitative diagnostic probes

**These 29 commands were written by hand for this analysis. They are qualitative diagnostic probes, not an independent statistical test set, and support no claim about generalization.** `phrase_in_dataset` flags probes whose normalized phrase also appears in the workbook (several example probes requested in the brief, such as `tell me a joke`, exist in the unknown set). `AMBIGUOUS` marks commands with no single correct intent.

| text | expected | final_intent | predicted_intent | max_confidence | matches_expected | phrase_in_dataset | probe_purpose |
|---|---|---|---|---|---|---|---|
| send fifty dollars to Jone | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.827 | yes | no | number written as words |
| transfer $20 to Mere | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.819 | yes | no | currency symbol |
| SEND $500 TO JONE!!! | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.865 | yes | no | capitalisation and punctuation |
| snd 50 2 jone | TRANSFER_MONEY | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.213 | no | no | SMS-style spelling |
| Bula, send $30 to Jone vinaka | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.879 | yes | no | Fijian greeting/thanks code-switching |
| pay Jone back $40 | TRANSFER_MONEY | BILL_PAY | BILL_PAY | 0.452 | no | no | 'pay' + person: TRANSFER vs BILL_PAY risk |
| pay my power bill | BILL_PAY | BILL_PAY | BILL_PAY | 0.885 | yes | no | synonym for electricity |
| pay the water authority bill | BILL_PAY | BILL_PAY | BILL_PAY | 0.908 | yes | no | utility name variant |
| show my last five transactions | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.634 | yes | no | number as word |
| Jone sent me $50 did it arrive | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.389 | yes | no | incoming transfer; shares words with TRANSFER |
| what's my balance | CHECK_BALANCE | CHECK_BALANCE | CHECK_BALANCE | 0.543 | yes | yes | contraction |
| how much is left in my wallet | CHECK_BALANCE | CHECK_BALANCE | CHECK_BALANCE | 0.970 | yes | no | paraphrase |
| how much did I spend this week | CHECK_SPENDING | CHECK_SPENDING | CHECK_SPENDING | 0.854 | yes | no | stretch intent |
| what did I spend the most on last month | CHECK_SPENDING | CHECK_SPENDING | CHECK_SPENDING | 0.923 | yes | no | stretch intent paraphrase |
| cash out 100 dollars at the agent | WITHDRAW_MONEY | WITHDRAW_MONEY | WITHDRAW_MONEY | 0.745 | yes | no | agent cash-out phrasing |
| add Mere to my contacts | MANAGE_CONTACTS | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.680 | yes | no | prompt example |
| remove Jone from my saved payees | MANAGE_CONTACTS | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.599 | yes | no | payee vocabulary overlaps BILL_PAY |
| show me my family savings group | VIEW_GROUP | VIEW_GROUP | VIEW_GROUP | 0.839 | yes | no | group paraphrase |
| who has paid into the church group | VIEW_GROUP | VIEW_GROUP | VIEW_GROUP | 0.929 | yes | no | 'paid' overlaps BILL_PAY |
| I think someone stole my pin | HELP_SUPPORT | HELP_SUPPORT | HELP_SUPPORT | 0.720 | yes | no | security issue |
| my transfer didn't go through | HELP_SUPPORT | HELP_SUPPORT | HELP_SUPPORT | 0.599 | yes | no | 'transfer' keyword but support intent |
| how do I send money | HELP_SUPPORT | TRANSFER_MONEY | TRANSFER_MONEY | 0.293 | no | no | how-to question; debatable vs TRANSFER_MONEY |
| tell me a joke | UNKNOWN_INTENT | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.207 | no | yes | out of scope |
| who won the rugby | UNKNOWN_INTENT | VIEW_GROUP | VIEW_GROUP | 0.309 | no | yes | out of scope, local topic |
| what is the weather in Suva | UNKNOWN_INTENT | CHECK_BALANCE | CHECK_BALANCE | 0.339 | no | no | out of scope |
| book a flight to Nadi | UNKNOWN_INTENT | TRANSFER_MONEY | TRANSFER_MONEY | 0.433 | no | no | out of scope, transactional-sounding |
| send me a pizza | UNKNOWN_INTENT | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.288 | no | no | out of scope but uses 'send' |
| send money and pay my bill | AMBIGUOUS | BILL_PAY | BILL_PAY | 0.741 | n/a | no | multi-intent command |
| don't send money to Jone | AMBIGUOUS | TRANSFER_MONEY | TRANSFER_MONEY | 0.799 | n/a | no | negation |

Probes with a definite expected label matching the final (post-rejection) output: 19/27 (illustrative only). Out-of-scope probes accepted as a supported intent: 5/5.

## 10. Takeaways

1. Most held-out errors trace to unseen vocabulary in a few repeated paraphrase templates.
2. Confidence is low across the board (under-confident model), so confidence-based rejection cannot separate in-scope from out-of-scope requests.
3. BoW ignores word order and negation (`don't send money to Jone` → TRANSFER_MONEY).
4. Explicit TRANSFER_MONEY ↔ BILL_PAY confusion is rare, but out-of-scope requests do get routed to money-moving intents.
