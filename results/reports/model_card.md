# Model Card — Sole Smart Actions Intent Classifier (BoW + Logistic Regression)

## Model details

- **Name / version:** `bow_logreg` tuned, v1.0.0 (`models/bow_logreg_v1_tuned_v1.0.0.joblib`)
- **Type:** bag-of-words (`ngram(1,1) C=0.1 counts`) + multinomial logistic regression (scikit-learn 1.9.1), confidence-threshold rejection at 0.1957.
- **Size:** 250 vocabulary terms, 2,259 parameters, 23.3 KB.
- **Developed for:** MIT AI Edge course project (academic baseline). Created by the Sole Smart Actions team.
- **Artifact contents:** sklearn Pipeline, canonical class order, rejection threshold, metadata (dataset hash, package versions, params). Load with `sole_intent.train.load_bundle`.

## Intended use

- Research baseline for classifying short digital-wallet commands into 9 intents, and a reference point for CNN/LSTM/transformer comparisons using the shared evaluation framework.
- **Out of scope:** executing payments or any live financial action; production deployment; languages other than English; safety-critical decisions without human or explicit-confirmation controls.

## Training data

`Sole_Smart_Actions_Intent_Training_v4_Cleaned.xlsx` (version `v4_cleaned`, SHA-256 `708e8f79ca47c0df689c97da02b21017607e2f55e50b30dad4cb0d9cc7033bda`): 2,160 synthetic training utterances, 240 per intent. Validation (270) was used for model and threshold selection. Test (270) and 300 unknown requests were used once for final evaluation.

## Metrics (held-out test unless stated)

| Metric | Value |
|---|---|
| Accuracy | 0.793 |
| Macro F1 (95% phrase-clustered CI) | 0.796 [0.712, 0.864] |
| Weighted F1 | 0.796 |
| Macro ROC-AUC (OvR) | 0.954 |
| Log loss | 0.985 |
| Core-8 macro F1 (9-class model) | 0.804 |
| Leakage-controlled test macro F1 | 0.774 |
| Train / validation macro F1 | 0.987 / 0.735 |
| Known-intent coverage at threshold | 90.7% |
| Unknown rejection rate / false acceptance | 14.7% / 85.3% |
| Single-request latency median / P95 (CPU) | 0.144 / 0.173 ms |

## Limitations and risks

- Trained and evaluated on synthetic English templates. Expect lower accuracy on real Fijian customer language (code-switching with Fijian / Fiji Hindi, slang, typos, speech-to-text errors).
- Unseen words are ignored. Paraphrases built on new vocabulary are frequently misclassified.
- Ignores word order and negation, so it cannot tell payer from payee.
- Weak out-of-scope rejection: 85.3% of unsupported requests were accepted, some as money-moving intents.
- Probabilities are under-confident (poorly calibrated).
- Logistic-regression coefficients (`13_top_features.png`) show associations between n-grams and predicted classes. They are not causal explanations.

## Ethical and safety considerations

Misrouting a financial request can cause monetary harm. Downstream systems must confirm amount, payee and action with the user before any money moves, and should hand unclear requests to a help or clarification flow. No personal or real customer data was used.
