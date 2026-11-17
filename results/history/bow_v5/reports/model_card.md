# Model Card — Sole Smart Actions Intent Classifier (BoW + Logistic Regression)

## Model details

- **Name / version:** `bow_logreg` tuned, v1.0.0 (`models_v5/bow_logreg_v5_tuned_v1.0.0.joblib`)
- **Type:** bag-of-words (`ngram(1,2) C=10 binary`) + multinomial logistic regression (scikit-learn 1.9.1), confidence-threshold rejection at 0.3415.
- **Size:** 2584 vocabulary terms, 23,265 parameters, 222.1 KB.
- **Developed for:** MIT AI Edge course project (academic baseline). Created by the Sole Smart Actions team.
- **Artifact contents:** sklearn Pipeline, canonical class order, rejection threshold, metadata (dataset hash, package versions, params). Load with `sole_intent.train.load_bundle`.

## Intended use

- Research baseline for classifying short digital-wallet commands into 9 intents, and a reference point for CNN/LSTM/transformer comparisons using the shared evaluation framework.
- **Out of scope:** executing payments or any live financial action; production deployment; languages other than English; safety-critical decisions without human or explicit-confirmation controls.

## Training data

`Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx` (version `v5_deduplicated`, SHA-256 `8dc9c89cbb8af41fd7cb25b2841e2f1a0c986641c4a6108c2f804d09f9345516`): 2,160 synthetic training utterances, 240 per intent. Validation (270) was used for model and threshold selection. Test (270) and 300 unknown requests were used once for final evaluation.

## Metrics (held-out test unless stated)

| Metric | Value |
|---|---|
| Accuracy | 0.937 |
| Macro F1 (95% phrase-clustered CI) | 0.937 [0.905, 0.964] |
| Weighted F1 | 0.937 |
| Macro ROC-AUC (OvR) | 0.995 |
| Log loss | 0.247 |
| Core-8 macro F1 (9-class model) | 0.937 |
| Leakage-controlled test macro F1 | 0.926 |
| Train / validation macro F1 | 1.000 / 0.917 |
| Known-intent coverage at threshold | 96.3% |
| Unknown rejection rate / false acceptance | 29.3% / 70.7% |
| Single-request latency median / P95 (CPU) | 0.280 / 0.444 ms |

## Limitations and risks

- Trained and evaluated on synthetic English templates. Expect lower accuracy on real Fijian customer language (code-switching with Fijian / Fiji Hindi, slang, typos, speech-to-text errors).
- Unseen words are ignored. Paraphrases built on new vocabulary are frequently misclassified.
- Ignores word order and negation, so it cannot tell payer from payee.
- Weak out-of-scope rejection: 70.7% of unsupported requests were accepted, some as money-moving intents.
- Probabilities are under-confident (poorly calibrated).
- Logistic-regression coefficients (`13_top_features.png`) show associations between n-grams and predicted classes. They are not causal explanations.

## Ethical and safety considerations

Misrouting a financial request can cause monetary harm. Downstream systems must confirm amount, payee and action with the user before any money moves, and should hand unclear requests to a help or clarification flow. No personal or real customer data was used.
