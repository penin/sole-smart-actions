# Error Analysis — BoW + Logistic Regression (selected model)

_Generated 2026-11-12T02:55:57+00:00. Threshold for rejection: 0.3415. Row-level predictions with all class probabilities: `results_v5/predictions/*_predictions.csv`. Tables below are also saved as `results_v5/metrics/error_*.csv`._

## 1. Most common confusions

| split | actual_intent | predicted_intent | count | unique_phrases | example |
|---|---|---|---|---|---|
| test | TRANSFER_MONEY | HELP_SUPPORT | 7 | 7 | I owe joeli 100 |
| test | CHECK_SPENDING | VIEW_TRANSACTIONS | 2 | 2 | Show expenditure summary |
| test | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 2 | 2 | Show beneficiaries |
| test | HELP_SUPPORT | BILL_PAY | 1 | 1 | I have a complaint |
| test | HELP_SUPPORT | MANAGE_CONTACTS | 1 | 1 | Need customer service |
| test | TRANSFER_MONEY | WITHDRAW_MONEY | 1 | 1 | I owe mary $50 |
| test | VIEW_TRANSACTIONS | CHECK_BALANCE | 1 | 1 | Where is my statement this time |
| test | VIEW_TRANSACTIONS | HELP_SUPPORT | 1 | 1 | Where is my statement on my account |
| test | VIEW_TRANSACTIONS | MANAGE_CONTACTS | 1 | 1 | Where is my statement |
| validation | VIEW_TRANSACTIONS | TRANSFER_MONEY | 11 | 11 | Find my transfer to adi |
| validation | HELP_SUPPORT | CHECK_BALANCE | 4 | 4 | My money is missing |
| validation | BILL_PAY | TRANSFER_MONEY | 3 | 3 | I need to pay internet |
| validation | VIEW_TRANSACTIONS | VIEW_GROUP | 2 | 2 | Open statement |
| validation | CHECK_BALANCE | HELP_SUPPORT | 1 | 1 | Do i have 250 on my account |
| validation | VIEW_GROUP | CHECK_SPENDING | 1 | 1 | How much did i contribute |

Train confusions (in-sample):

_(none)_

## 2. Confident errors

Errors with confidence ≥ 0.9: 6 on validation/test. The selected model's highest held-out confidence is 1.000. Confident errors cannot be filtered by any reasonable threshold. Below: **errors that the rejection threshold would accept** (distinct phrases, highest confidence first):

| split | text | actual_intent | predicted_intent | max_confidence | top2_margin | oov_tokens |
|---|---|---|---|---|---|---|
| validation | How much did i contribute | VIEW_GROUP | CHECK_SPENDING | 0.979 | 0.971 | contribute |
| validation | Find my transfer to ana | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.946 | 0.913 |  |
| validation | Find my transfer to sera | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.929 | 0.886 |  |
| validation | Find my transfer to mary | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.916 | 0.864 |  |
| validation | Find my transfer to peni | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.906 | 0.848 |  |
| validation | Find my transfer to ravi | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.905 | 0.847 |  |
| validation | Find my transfer to roshni | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.879 | 0.807 |  |
| validation | Find my transfer to mere | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.877 | 0.801 |  |
| validation | Find my transfer to adi | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.874 | 0.799 |  |
| validation | Find my transfer to ben | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.857 | 0.772 |  |
| validation | Find my transfer to raj | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.851 | 0.761 |  |
| validation | Find my transfer to asha | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.839 | 0.743 |  |
| test | Show beneficiaries | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.748 | 0.675 | beneficiaries |
| test | Show expenditure summary | CHECK_SPENDING | VIEW_TRANSACTIONS | 0.694 | 0.614 |  |
| test | Show beneficiaries quickly | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.692 | 0.594 | beneficiaries |
| validation | I need to pay telecom | BILL_PAY | TRANSFER_MONEY | 0.666 | 0.344 |  |
| validation | Do i have 250 on my account | CHECK_BALANCE | HELP_SUPPORT | 0.574 | 0.195 |  |
| validation | I need to pay phone | BILL_PAY | TRANSFER_MONEY | 0.547 | 0.112 |  |
| validation | I need to pay internet | BILL_PAY | TRANSFER_MONEY | 0.524 | 0.060 |  |
| validation | My money is missing now | HELP_SUPPORT | CHECK_BALANCE | 0.518 | 0.350 |  |

## 3. Low-confidence correct predictions (would be falsely rejected)

14 distinct correct validation/test phrases fall below the threshold:

| split | text | actual_intent | predicted_intent | max_confidence | top2_margin | oov_tokens |
|---|---|---|---|---|---|---|
| validation | Look up mere | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.225 | 0.000 | look |
| validation | What's my wallet total | CHECK_BALANCE | CHECK_BALANCE | 0.237 | 0.045 |  |
| test | View groupwealth now | VIEW_GROUP | VIEW_GROUP | 0.256 | 0.014 |  |
| validation | Look up ana | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.256 | 0.059 | look |
| validation | Look up adi | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.258 | 0.036 | look |
| validation | Look up sera | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.264 | 0.046 | look |
| validation | Look up savenaca | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.265 | 0.044 | look |
| validation | Look up raj | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.266 | 0.039 | look |
| test | View groupwealth | VIEW_GROUP | VIEW_GROUP | 0.266 | 0.043 |  |
| validation | Look up mika | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.274 | 0.066 | look |
| validation | Look up ravi | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.284 | 0.069 | look |
| validation | Look up tui | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.296 | 0.093 | look |
| validation | Look up priya | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.304 | 0.089 | look |
| validation | Look up jone | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.306 | 0.101 | look |

## 4. Ambiguous commands (top-2 margin < 0.1)

| split | text | actual_intent | predicted_intent | second_intent | top2_margin | correct |
|---|---|---|---|---|---|---|
| validation | Look up mere | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.000 | yes |
| test | I owe jone 80 | TRANSFER_MONEY | HELP_SUPPORT | TRANSFER_MONEY | 0.005 | no |
| test | Need customer service | HELP_SUPPORT | MANAGE_CONTACTS | HELP_SUPPORT | 0.006 | no |
| test | I owe mary $50 | TRANSFER_MONEY | WITHDRAW_MONEY | TRANSFER_MONEY | 0.014 | no |
| test | View groupwealth now | VIEW_GROUP | VIEW_GROUP | HELP_SUPPORT | 0.014 | yes |
| validation | Open statement if possible | VIEW_TRANSACTIONS | VIEW_GROUP | HELP_SUPPORT | 0.015 | no |
| test | I owe mika 5 | TRANSFER_MONEY | HELP_SUPPORT | TRANSFER_MONEY | 0.026 | no |
| test | I owe joeli 100 | TRANSFER_MONEY | HELP_SUPPORT | WITHDRAW_MONEY | 0.031 | no |
| validation | Look up adi | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.036 | yes |
| validation | Look up raj | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.039 | yes |
| test | I owe mere 50 | TRANSFER_MONEY | HELP_SUPPORT | WITHDRAW_MONEY | 0.041 | no |
| test | I owe adi 250 | TRANSFER_MONEY | HELP_SUPPORT | WITHDRAW_MONEY | 0.042 | no |
| test | View groupwealth | VIEW_GROUP | VIEW_GROUP | VIEW_TRANSACTIONS | 0.043 | yes |
| validation | Look up savenaca | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.044 | yes |
| test | I owe kumar 40 | TRANSFER_MONEY | HELP_SUPPORT | MANAGE_CONTACTS | 0.045 | no |
| validation | What's my wallet total | CHECK_BALANCE | CHECK_BALANCE | CHECK_SPENDING | 0.045 | yes |
| validation | Look up sera | MANAGE_CONTACTS | MANAGE_CONTACTS | CHECK_SPENDING | 0.046 | yes |
| validation | Do i have 10 | CHECK_BALANCE | CHECK_BALANCE | HELP_SUPPORT | 0.047 | yes |
| validation | Do i have 150 | CHECK_BALANCE | CHECK_BALANCE | HELP_SUPPORT | 0.050 | yes |
| validation | Do i have 80 | CHECK_BALANCE | CHECK_BALANCE | HELP_SUPPORT | 0.051 | yes |

## 5. Short commands (≤ 2 tokens) and out-of-vocabulary terms

| split | n | short_n | short_accuracy | longer_accuracy | rows_with_oov | share_rows_with_oov | oov_accuracy | in_vocab_accuracy | mean_oov_token_share |
|---|---|---|---|---|---|---|---|---|---|
| train | 2,160 | 71 | 1.000 | 1.000 | 0 | 0.000 | n/a | 1.000 | 0.000 |
| validation | 270 | 7 | 0.857 | 0.920 | 29 | 0.107 | 0.897 | 0.921 | 0.032 |
| test | 270 | 13 | 0.923 | 0.938 | 31 | 0.115 | 0.516 | 0.992 | 0.032 |

Short-command errors (validation/test):

| split | text | actual_intent | predicted_intent | max_confidence | top2_margin | oov_tokens |
|---|---|---|---|---|---|---|
| validation | Open statement | VIEW_TRANSACTIONS | VIEW_GROUP | 0.317 | 0.110 | statement |
| test | Show beneficiaries | MANAGE_CONTACTS | VIEW_TRANSACTIONS | 0.748 | 0.675 | beneficiaries |

Most frequent out-of-vocabulary tokens in validation/test (never seen in training):

| oov_token | held_out_occurrences |
|---|---|
| look | 11 |
| owe | 10 |
| using | 8 |
| statement | 5 |
| talk | 5 |
| recent | 4 |
| customer | 4 |
| service | 4 |
| average | 2 |
| daily | 2 |
| complaint | 2 |
| kesa | 2 |
| beneficiaries | 2 |
| 90 | 2 |
| contribute | 1 |
| prepare | 1 |
| manage | 1 |

88.2% of test errors contain a word never seen in training — out-of-vocabulary words explain most errors. The vectorizer silently drops unseen words; e.g. in `Need customer service` the word(s) `customer service` are ignored, leaving the model to guess MANAGE_CONTACTS (actual HELP_SUPPORT).

## 6. Word-order blindness

| text_a | pred_a | conf_a | text_b | pred_b | conf_b | same_bag_of_words | max_abs_prob_diff |
|---|---|---|---|---|---|---|---|
| send money to jone | TRANSFER_MONEY | 0.998 | jone send money to | TRANSFER_MONEY | 0.995 | yes | 0.003 |
| jone sent me money | VIEW_TRANSACTIONS | 0.780 | i sent jone money | HELP_SUPPORT | 0.191 | no | 0.634 |
| pay bill then check balance | BILL_PAY | 0.848 | check balance then pay bill | BILL_PAY | 0.848 | yes | 0.000 |
| transfer from mere to jone | TRANSFER_MONEY | 0.994 | transfer from jone to mere | TRANSFER_MONEY | 0.988 | yes | 0.006 |
| did i send money to mere | TRANSFER_MONEY | 0.995 | i did send money to mere | TRANSFER_MONEY | 0.989 | yes | 0.006 |

Pairs with the same bag of words receive **identical** probabilities (max |Δp| = 0). The model cannot tell who pays whom (`transfer from mere to jone` vs `from jone to mere`) or the order of a multi-step request. Payer/payee direction must come from a slot-filling step, not from this classifier.

## 7. Train/test leakage

| split | leakage_flag | n | accuracy |
|---|---|---|---|
| validation | no | 247 | 0.911 |
| validation | yes | 23 | 1.000 |
| test | no | 231 | 0.926 |
| test | yes | 39 | 1.000 |

Test rows that repeat a training phrase (exact or number-masked):

| text | actual_intent | predicted_intent | max_confidence | correct |
|---|---|---|---|---|
| Display my wallet balance on 15 September | CHECK_BALANCE | CHECK_BALANCE | 0.998 | yes |
| Show last 5 transactions | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.985 | yes |
| Please settle my EFL invoice for $175 | BILL_PAY | BILL_PAY | 0.999 | yes |
| Pay 70 dollars toward my water supplier bill | BILL_PAY | BILL_PAY | 0.999 | yes |
| Settle $350 owed to internet provider | BILL_PAY | BILL_PAY | 0.999 | yes |
| Show last 5 transactions on my account | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.996 | yes |
| Give me an expenditure breakdown on 27 May | CHECK_SPENDING | CHECK_SPENDING | 0.995 | yes |
| Help me pay $250 to internet provider | BILL_PAY | BILL_PAY | 0.999 | yes |
| Give me an expenditure breakdown on 25 December | CHECK_SPENDING | CHECK_SPENDING | 0.994 | yes |
| Show incoming transfers on 26 May | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.997 | yes |
| What is my available balance on 15 October? | CHECK_BALANCE | CHECK_BALANCE | 1.000 | yes |
| Settle $300 owed to internet provider | BILL_PAY | BILL_PAY | 0.999 | yes |
| How much cash is in Sole on 12 April? | CHECK_BALANCE | CHECK_BALANCE | 0.997 | yes |
| Show incoming transfers on 9 December | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.998 | yes |
| Pay my electricity provider bill of $90 | BILL_PAY | BILL_PAY | 1.000 | yes |
| Tell me how much I paid out on 27 July | CHECK_SPENDING | CHECK_SPENDING | 0.997 | yes |
| Can you check my account balance on 18 April? | CHECK_BALANCE | CHECK_BALANCE | 0.999 | yes |
| Show outgoing payments on 3 March | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.996 | yes |
| Show outgoing payments on 23 March | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.996 | yes |
| Show the amount available to spend on 15 September | CHECK_BALANCE | CHECK_BALANCE | 0.989 | yes |
| Help me take out 60 dollars | WITHDRAW_MONEY | WITHDRAW_MONEY | 0.999 | yes |
| Withdraw $70 from my wallet | WITHDRAW_MONEY | WITHDRAW_MONEY | 1.000 | yes |
| Settle $5 owed to mobile provider | BILL_PAY | BILL_PAY | 0.998 | yes |
| Show money received on 20 March | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.995 | yes |
| Give me the transaction list for on 6 October | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.998 | yes |
| Display my wallet balance on 26 November | CHECK_BALANCE | CHECK_BALANCE | 0.999 | yes |
| List contributions in family on 5 March | VIEW_GROUP | VIEW_GROUP | 0.995 | yes |
| Show the amount available to spend on 5 June | CHECK_BALANCE | CHECK_BALANCE | 0.993 | yes |
| Request a withdrawal of $35 | WITHDRAW_MONEY | WITHDRAW_MONEY | 0.984 | yes |
| Withdraw $100 from my wallet | WITHDRAW_MONEY | WITHDRAW_MONEY | 1.000 | yes |
| I need to collect $220 in cash | WITHDRAW_MONEY | WITHDRAW_MONEY | 0.996 | yes |
| Tell me my current balance on 20 August | CHECK_BALANCE | CHECK_BALANCE | 0.999 | yes |
| Do I have money in my Sole wallet on 4 December? | CHECK_BALANCE | CHECK_BALANCE | 0.999 | yes |
| Display my latest transaction history on 13 September | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.996 | yes |
| How do I withdraw 80 dollars? | WITHDRAW_MONEY | WITHDRAW_MONEY | 0.996 | yes |
| Take $60 out of my Sole account | WITHDRAW_MONEY | WITHDRAW_MONEY | 0.997 | yes |
| Show money received on 8 February | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.994 | yes |
| Show incoming transfers on 3 November | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.996 | yes |
| List my wallet transactions on 27 October | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.995 | yes |

## 8. Financial-safety view

TRANSFER_MONEY ↔ BILL_PAY confusions:

| split | actual | predicted | count | accepted_at_threshold |
|---|---|---|---|---|
| train | TRANSFER_MONEY | BILL_PAY | 0 | 0 |
| train | BILL_PAY | TRANSFER_MONEY | 0 | 0 |
| validation | TRANSFER_MONEY | BILL_PAY | 0 | 0 |
| validation | BILL_PAY | TRANSFER_MONEY | 3 | 3 |
| test | TRANSFER_MONEY | BILL_PAY | 0 | 0 |
| test | BILL_PAY | TRANSFER_MONEY | 0 | 0 |

Training TRANSFER_MONEY rows predicted BILL_PAY (in-sample):

_(none)_

A transfer to a person mistaken for a bill payment (or vice versa) is the most consequential confusion for a wallet. A payee/biller slot check should gate both intents.

Supported requests wrongly routed **and accepted** into a money-moving intent (validation/test):

| split | text | actual_intent | predicted_intent | max_confidence | top2_margin | oov_tokens |
|---|---|---|---|---|---|---|
| validation | Find my transfer to adi | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.874 | 0.799 |  |
| validation | I need to pay internet | BILL_PAY | TRANSFER_MONEY | 0.524 | 0.060 |  |
| validation | Find my transfer to raj | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.851 | 0.761 |  |
| validation | Find my transfer to asha | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.839 | 0.743 |  |
| validation | Find my transfer to ben | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.857 | 0.772 |  |
| validation | Find my transfer to peni | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.906 | 0.848 |  |
| validation | Find my transfer to roshni | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.879 | 0.807 |  |
| validation | I need to pay phone | BILL_PAY | TRANSFER_MONEY | 0.547 | 0.112 |  |
| validation | I need to pay telecom | BILL_PAY | TRANSFER_MONEY | 0.666 | 0.344 |  |
| validation | Find my transfer to mary | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.916 | 0.864 |  |
| validation | Find my transfer to ravi | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.905 | 0.847 |  |
| validation | Find my transfer to ana | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.946 | 0.913 |  |
| validation | Find my transfer to sera | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.929 | 0.886 |  |
| validation | Find my transfer to mere | VIEW_TRANSACTIONS | TRANSFER_MONEY | 0.877 | 0.801 |  |
| test | I have a complaint | HELP_SUPPORT | BILL_PAY | 0.408 | 0.078 | complaint |
| test | I owe mary $50 | TRANSFER_MONEY | WITHDRAW_MONEY | 0.372 | 0.014 | owe |

Unsupported (UNKNOWN_INTENT) requests accepted as money-moving intents (distinct phrases, 23):

| text | final_intent | max_confidence |
|---|---|---|
| Set a reminder to water the plants on 6 March | BILL_PAY | 0.736 |
| Set a reminder to water the plants on 26 March | BILL_PAY | 0.731 |
| Set a reminder to water the plants on 22 July | BILL_PAY | 0.686 |
| Set a reminder to water the plants on 28 December | BILL_PAY | 0.682 |
| Set a reminder to water the plants on 28 June | BILL_PAY | 0.680 |
| Set a reminder to water the plants on 21 August | BILL_PAY | 0.676 |
| Set a reminder to water the plants on 10 February | BILL_PAY | 0.675 |
| Buy me a phone | BILL_PAY | 0.596 |
| Can you book a flight to Nadi? | TRANSFER_MONEY | 0.547 |
| Can you book a flight to Suva? | TRANSFER_MONEY | 0.547 |
| Can you book a flight to Tokyo? | TRANSFER_MONEY | 0.547 |
| Can you book a flight to London? | TRANSFER_MONEY | 0.547 |
| Can you book a flight to Lautoka? | TRANSFER_MONEY | 0.547 |
| Can you book a flight to Apia? | TRANSFER_MONEY | 0.547 |
| Can you book a flight to Singapore? | TRANSFER_MONEY | 0.547 |
| Can you book a flight to Sydney? | TRANSFER_MONEY | 0.547 |
| Can you book a flight to Paris? | TRANSFER_MONEY | 0.547 |
| Buy me a phone if possible | BILL_PAY | 0.490 |
| Buy me a phone now | BILL_PAY | 0.443 |
| Buy me a phone please | BILL_PAY | 0.438 |

Most confident accepted unknowns (all intents):

| text | final_intent | max_confidence |
|---|---|---|
| Book a taxi right now on my account | HELP_SUPPORT | 0.872 |
| Show me videos of turtle | VIEW_TRANSACTIONS | 0.860 |
| Show me videos of cat | VIEW_TRANSACTIONS | 0.860 |
| Show me videos of parrot | VIEW_TRANSACTIONS | 0.860 |
| Show me videos of dog | VIEW_TRANSACTIONS | 0.860 |
| Show me videos of horse | VIEW_TRANSACTIONS | 0.860 |
| Find a cinema on the app | VIEW_TRANSACTIONS | 0.829 |
| Find a hotel on the app | VIEW_TRANSACTIONS | 0.829 |
| Find a curry recipe on the app | VIEW_TRANSACTIONS | 0.829 |
| Set an alarm if possible | HELP_SUPPORT | 0.763 |
| Order food please if possible | HELP_SUPPORT | 0.758 |
| Set a reminder to water the plants on 6 March | BILL_PAY | 0.736 |
| Set a reminder to water the plants on 26 March | BILL_PAY | 0.731 |
| Order food please quickly | HELP_SUPPORT | 0.725 |
| Buy me a phone quickly at the moment | HELP_SUPPORT | 0.718 |

Any production use would need an explicit confirmation step (amount, payee, biller) before a money-moving action. Classifier confidence alone is insufficient.

## 9. Qualitative diagnostic probes

**These 29 commands were written by hand for this analysis. They are qualitative diagnostic probes, not an independent statistical test set, and support no claim about generalization.** `phrase_in_dataset` flags probes whose normalized phrase also appears in the workbook (3 of 29). `AMBIGUOUS` marks commands with no single correct intent.

| text | expected | final_intent | predicted_intent | max_confidence | matches_expected | phrase_in_dataset | probe_purpose |
|---|---|---|---|---|---|---|---|
| send fifty dollars to Jone | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.995 | yes | no | number written as words |
| transfer $20 to Mere | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.994 | yes | no | currency symbol |
| SEND $500 TO JONE!!! | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.998 | yes | no | capitalisation and punctuation |
| snd 50 2 jone | TRANSFER_MONEY | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.361 | no | no | SMS-style spelling |
| Bula, send $30 to Jone vinaka | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.999 | yes | no | Fijian greeting/thanks code-switching |
| pay Jone back $40 | TRANSFER_MONEY | TRANSFER_MONEY | TRANSFER_MONEY | 0.665 | yes | no | 'pay' + person: TRANSFER vs BILL_PAY risk |
| pay my power bill | BILL_PAY | BILL_PAY | BILL_PAY | 0.997 | yes | no | synonym for electricity |
| pay the water authority bill | BILL_PAY | BILL_PAY | BILL_PAY | 0.999 | yes | no | utility name variant |
| show my last five transactions | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.953 | yes | no | number as word |
| Jone sent me $50 did it arrive | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.441 | yes | no | incoming transfer; shares words with TRANSFER |
| what's my balance | CHECK_BALANCE | CHECK_BALANCE | CHECK_BALANCE | 0.974 | yes | yes | contraction |
| how much is left in my wallet | CHECK_BALANCE | CHECK_BALANCE | CHECK_BALANCE | 1.000 | yes | no | paraphrase |
| how much did I spend this week | CHECK_SPENDING | CHECK_SPENDING | CHECK_SPENDING | 0.997 | yes | no | stretch intent |
| what did I spend the most on last month | CHECK_SPENDING | CHECK_SPENDING | CHECK_SPENDING | 0.997 | yes | no | stretch intent paraphrase |
| cash out 100 dollars at the agent | WITHDRAW_MONEY | WITHDRAW_MONEY | WITHDRAW_MONEY | 0.992 | yes | no | agent cash-out phrasing |
| add Mere to my contacts | MANAGE_CONTACTS | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.994 | yes | no | prompt example |
| remove Jone from my saved payees | MANAGE_CONTACTS | MANAGE_CONTACTS | MANAGE_CONTACTS | 0.998 | yes | no | payee vocabulary overlaps BILL_PAY |
| show me my family savings group | VIEW_GROUP | VIEW_GROUP | VIEW_GROUP | 0.995 | yes | no | group paraphrase |
| who has paid into the church group | VIEW_GROUP | VIEW_GROUP | VIEW_GROUP | 0.999 | yes | no | 'paid' overlaps BILL_PAY |
| I think someone stole my pin | HELP_SUPPORT | HELP_SUPPORT | HELP_SUPPORT | 0.984 | yes | no | security issue |
| my transfer didn't go through | HELP_SUPPORT | HELP_SUPPORT | HELP_SUPPORT | 0.600 | yes | no | 'transfer' keyword but support intent |
| how do I send money | HELP_SUPPORT | TRANSFER_MONEY | TRANSFER_MONEY | 0.814 | no | no | how-to question; debatable vs TRANSFER_MONEY |
| tell me a joke | UNKNOWN_INTENT | UNKNOWN_INTENT | HELP_SUPPORT | 0.238 | yes | yes | out of scope |
| who won the rugby | UNKNOWN_INTENT | VIEW_GROUP | VIEW_GROUP | 0.491 | no | yes | out of scope, local topic |
| what is the weather in Suva | UNKNOWN_INTENT | CHECK_BALANCE | CHECK_BALANCE | 0.385 | no | no | out of scope |
| book a flight to Nadi | UNKNOWN_INTENT | TRANSFER_MONEY | TRANSFER_MONEY | 0.389 | no | no | out of scope, transactional-sounding |
| send me a pizza | UNKNOWN_INTENT | VIEW_TRANSACTIONS | VIEW_TRANSACTIONS | 0.636 | no | no | out of scope but uses 'send' |
| send money and pay my bill | AMBIGUOUS | BILL_PAY | BILL_PAY | 0.893 | n/a | no | multi-intent command |
| don't send money to Jone | AMBIGUOUS | TRANSFER_MONEY | TRANSFER_MONEY | 0.998 | n/a | no | negation |

Probes with a definite expected label matching the final (post-rejection) output: 21/27 (illustrative only). Out-of-scope probes accepted as a supported intent: 4/5.

## 10. Takeaways

1. 88.2% of held-out test errors involve vocabulary unseen in training.
2. Confidence is low across the board (under-confident model), so confidence-based rejection cannot cleanly separate in-scope from out-of-scope requests.
3. BoW ignores word order and negation (`don't send money to Jone` → TRANSFER_MONEY).
4. TRANSFER_MONEY ↔ BILL_PAY confusions on held-out data: 3; out-of-scope requests routed to money-moving intents: 23 distinct phrases.
