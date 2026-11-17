# Sole Smart Actions: Intent Classification with TF-IDF + Logistic Regression

**Audience:** data scientists / ML engineers on the Sole wallet team (Fiji).
**Question:** does a richer text representation (TF-IDF, and TF-IDF with character n-grams) route natural-language wallet commands ("Send $50 to Jone") to the right workflow better than the team's Bag-of-Words benchmark?

## Task and data
- **Prediction task:** short English command → one of 9 in-scope intents: `TRANSFER_MONEY, CHECK_BALANCE, VIEW_TRANSACTIONS, WITHDRAW_MONEY, BILL_PAY, HELP_SUPPORT, MANAGE_CONTACTS, VIEW_GROUP, CHECK_SPENDING` (stretch). A confidence gate sends uncertain commands to a fallback (clarify / Help & Support).
- **Data:** `data/Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx`, sheet *Training Data*. 3,000 synthetic, de-duplicated commands (5.7 words on average). Predefined splits: train 2,160 · validation 270 · test 270 (30 per intent) · 300 `UNKNOWN_INTENT` requests ("Book a taxi"), used **only** to test rejection.
- **Protocol:** hyperparameters and confidence threshold chosen on validation only (threshold keeps 95% of valid commands). The test set is scored once. Seed = 42.

## Models compared
| Model | Representation | Selected setting |
|---|---|---|
| Keyword rules | Ordered regex rules (stand-in for menu search) | none |
| BoW + LR (team benchmark) | Unigram counts | C = 0.5 |
| TF-IDF (word) + LR | Unigram TF-IDF, sublinear tf | C = 5 |
| **TF-IDF (word + char) + LR** | Word 1–2-gram + char_wb 2–5-gram TF-IDF | C = 10 |

## Results (held-out test set, n = 270)
| Model | Val macro-F1 | Test macro-F1 | Test macro-F1, 1 typo per command | Wrong screen* | Out-of-scope accepted* |
|---|---|---|---|---|---|
| Keyword rules | 0.77 | 0.83 | 0.67 | 7.0% | 2.3% |
| BoW + LR | 0.86 | 0.94 | 0.81 | 3.7% | 72.0% |
| TF-IDF (word) + LR | 0.89 | 0.97 | 0.90 | 3.0% | 64.3% |
| **TF-IDF (word + char) + LR** | **0.94** | **0.97** | **0.96** | **1.5%** | 47.3% |

\*After the confidence gate. *Wrong screen* = valid commands sent to the wrong workflow. *Out-of-scope accepted* = UNKNOWN requests that got past the gate.
The best model makes 7 test errors (BoW makes 17). None of them opens a money-moving workflow (transfer, bill pay, withdraw). Inference takes ~1 ms per command on CPU; the model file is 0.6 MB.

**Key findings**
1. **Representation matters.** TF-IDF beats raw counts. Character n-grams add the most under noise: with one typo per command, macro-F1 drops only 0.97 → 0.96, compared with 0.94 → 0.81 for BoW.
2. **The remaining errors are vocabulary gaps, not model failures.** "Where is my statement" is predicted as CHECK_BALANCE, "Show beneficiaries" as VIEW_TRANSACTIONS, and "I have a complaint" as CHECK_BALANCE. Each needs more labelled examples.
3. **A confidence gate alone does not catch out-of-scope requests.** 47% still pass the gate. In an exploratory run, adding 150 UNKNOWN training examples rejected all 150 held-out UNKNOWN requests and cut wrong-screen routing to 0.4%. That result is optimistic because the UNKNOWN set is templated (`results/exploratory_out_of_scope.csv`).
4. **Caveats.** The data is synthetic, and the splits share templates. 5-fold CV gives ≈0.99 and validation gives 0.86–0.94, so expect lower accuracy on real user text. TRANSFER_MONEY partly keys on recipient names (e.g., "litia", "tui"). Before deployment, test on names the model has never seen.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_experiments.py      # ~15 s on a laptop; writes everything in results/
python src/predict.py "Send $50 to Jone" "Book a taxi"
```

## Repository layout
```
data/      V5 labelled dataset (synthetic, no customer data)
src/       run_experiments.py (training + evaluation + figures), predict.py (demo)
results/   model_comparison.csv, routing_outcomes_per_100k.csv, per_class_f1.csv,
           test_errors_best_model.csv, hyperparameter_search.csv, figures/*.png, best_model.pkl
```

## Recommendation
Ship TF-IDF (word + char) + LR as the production baseline and as the benchmark to beat for the Transformer and DistilBERT experiments. Before launch, add a labelled UNKNOWN class and fill the vocabulary gaps above. Then re-tune the gate on real traffic, and always require user confirmation before any payment is executed.
