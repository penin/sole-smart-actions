# Team contributions

Attribution is taken from the original deliverables (README files, notebook headers and git configuration). Where a
source names only a first name or a handle, that is what is recorded here.

| Contribution | Original author | Evidence | Where it lives now |
|---|---|---|---|
| Dataset V4/V5 workbooks (synthetic commands, splits, replacement log) | Project team | workbook `README` sheet | `data/raw/` |
| Bag of Words + Logistic Regression baseline, evaluation framework (metrics, bootstrap CIs, leakage analysis, threshold study), V4 -> V5 rerun, first web demo | penin | this repository's git history | `src/sole_intent/`, `results/history/`, `demo/` |
| TF-IDF + Logistic Regression (word and word + char n-grams), keyword-rule benchmark, typo robustness, routing-outcome analysis, exploratory UNKNOWN training | Chelsea Cheng | `sole-smart-actions-intent.zip` (author confirmed by the project owner) | `src/sole_smart_actions/models/tfidf/`, `research/original_experiments/tfidf/` |
| Custom Transformer from scratch: GloVe embeddings, 36-configuration search, attention / GloVe ablations, temperature calibration, value-based gate | Felipe (`friosf`) | notebook header of `01_custom_transformer_v5.ipynb` | `src/sole_smart_actions/models/transformer/`, `research/original_experiments/custom_transformer/` |
| Fine-tuned DistilBERT: unfreezing strategies, learning-rate search, calibration, two-way and three-tier gates, cost sensitivity; majority and keyword benchmarks | Ricardo (`ricardomanjarresr`) | notebook header and README of `01_sole_intent_classifier_v5.ipynb` (header lists "Team: Ricardo & Felipe") | `src/sole_smart_actions/models/distilbert/`, `research/original_experiments/distilbert/` |
| Integration into one repository: shared data validation, common model interface, unified evaluation and thresholds, reproducibility tooling, four-model demo | Integration work in this repository (git history) | commits after `1cd668e` | `src/sole_smart_actions/`, `scripts/`, `demo/`, `docs/` |

The two neural notebooks credit their structure and code to the MIT 15.071 *The AI Edge* lectures 10-12
(© 2026 Alexandre Jacquillat). GloVe embeddings: Pennington, Socher & Manning (2014). DistilBERT: Sanh et al. (2019),
distributed by Hugging Face.
