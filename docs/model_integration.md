# How the four implementations were integrated

This document records, model by model, what was kept from each original implementation and what was changed to
put all four models under one protocol. The originals are preserved unchanged (apart from removed personal paths) in
`research/original_experiments/`.

## Shared protocol (applies to every model)

| Rule | Implementation |
|---|---|
| One dataset | `data/raw/Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx`, SHA-256 pinned in `config/common.yaml`; `scripts.validate_data` runs 14 checks and stops on any failure |
| Fixed splits | the workbook's `split` column: train 2,160 / validation 270 / test 270 / evaluation_only 300; no model re-splits |
| Nine classes, fixed order | `config/common.yaml: labels`; every `predict_proba` returns columns in that order (checked in tests) |
| UNKNOWN_INTENT | evaluation only; never a training class (asserted when loading splits) |
| Training data | train split only (`trained_on: [train]` in every model card) |
| Model selection | validation split only |
| Default threshold | max probability; largest observed validation confidence keeping >= 95% of the 270 validation commands (`sole_intent.rejection.threshold_for_coverage`); no test or unknown labels |
| Metrics | one implementation (`sole_intent.evaluate.evaluate_predictions` + `sole_smart_actions.evaluation`) for all models; raw accuracy = correct / 270 before any threshold |
| Common interface | `IntentModel`: `fit`, `predict_proba`, `predict`, `save`, `load`, `get_model_metadata` (`src/sole_smart_actions/models/base.py`) |

## 1. Bag of Words + Logistic Regression (penin)

- **Kept:** everything. The adapter (`models/bow/model.py`) calls the original `sole_intent.train` functions:
  NFKC + lowercase + whitespace normalisation, `CountVectorizer` with the `$`-preserving token pattern, the 12-point grid
  (n-grams x C x binary), selection by validation macro F1 (ties: validation log loss), `lbfgs`, seed 42.
- **Verified:** the primary BoW model is bit-for-bit identical to the historical V5 model
  (`artifacts/history/bow_v5/`): maximum probability difference 0.0 on every split (`tests/test_ssa_models.py`).
- **History:** the V4 and V5 runs, their reports and the V4 threshold study are preserved under
  `results/history/` and `artifacts/history/` (see `results/history/README.md`).

## 2. TF-IDF + Logistic Regression (Chelsea Cheng)

- **Kept:** `build()` and the search space from `src/run_experiments.py`: word TF-IDF (1-1 or 1-2-grams) and
  word 1-2-gram + `char_wb` 2-4 / 2-5 / 3-5-gram TF-IDF via `FeatureUnion`, `sublinear_tf=True`, lowercase,
  C in {0.5, 1, 2, 5, 10, 20, 50}, `max_iter=5000`, seed 42, strip-only text cleaning, and the tie rule
  "best validation macro F1, then first in grid order".
- **Changed:** the original chose between the two representations by validation macro F1 and then 5-fold
  cross-validation on train + validation; the primary choice here uses validation macro F1 only. The default
  threshold uses the shared rule instead of `np.quantile` interpolation (0.404 vs the original 0.415).
- **Preserved as secondary analyses:** the typo-robustness test (`add_typos`, now run for every model:
  `robustness_typos.json`); the keyword-rule benchmark, BoW-in-TF-IDF benchmark and the exploratory
  "train on 150 UNKNOWN examples" run remain in `research/original_experiments/tfidf/`.
- **Result check:** same selected configuration (word + char 2-5, C = 10) and the same validation macro F1 (0.9378) and
  test accuracy (97.4%) as the original script.

## 3. Custom Transformer (Felipe, `friosf`)

- **Kept (verbatim apart from TensorFlow-only calls):** `PositionalEmbedding`, `TransformerBlock`,
  `build_transformer` (token + positional embeddings, masked multi-head self-attention block(s), mean over real tokens,
  64-unit ReLU head, softmax), GloVe 6B 100d initialisation (top-40k words + vocabulary, N(0, 0.1) for misses,
  zero padding row), Adam, sparse categorical cross-entropy, batch 32, up to 40 epochs with patience 10, best-epoch
  selection on validation accuracy (ties: log loss), the 36-configuration grid (heads 1/2/4 x blocks 1/2 x dropout
  0.1/0.2/0.3 x lr 3e-4/1e-3), `max_length=16`, `max_tokens=2000`, seed 1234, and the no-attention / no-GloVe
  ablations (`python -m scripts.train --model transformer --variant no_attention|no_glove`).
- **Changed:**
  1. nine classes (CHECK_SPENDING added; the original set `INCLUDE_STRETCH = False`), 270 test commands;
  2. the primary model is the best validation epoch of the train-only run; the original refit on
     train + validation is available as `--protocol trainval_refit` and reported separately;
  3. Keras 3 runs on the **PyTorch backend** (one deep-learning framework for the project; TensorFlow has no GPU
     support on native Windows). `keras.layers.TextVectorization` is TensorFlow-only, so it was replaced by
     `models/transformer/vectorizer.py`, which reproduces its defaults (`lower_and_strip_punctuation`, whitespace
     split, index 0 = padding, 1 = `[UNK]`); vocabulary frequency ties are broken alphabetically for determinism;
  4. GloVe is downloaded from a pinned Hugging Face revision and checked against a SHA-256 in `config/transformer.yaml`;
  5. the dollar-value gate tuned on 150 UNKNOWN examples is replaced by the shared 95%-coverage threshold; temperature
     scaling is reported as a secondary analysis (`secondary_analyses.json`);
  6. an input with no tokens at all (e.g. `""` or `"$!?"`, since punctuation is stripped) is fully masked, and the
     original masked mean pooling then returns NaN. At inference it is now encoded as a single `[UNK]` token. No V5
     command is affected (verified: published predictions unchanged).
- **Not ported:** the dollar-value business case (error prices are assumptions; kept in the original notebook).

## 4. Fine-tuned DistilBERT (Ricardo, `ricardomanjarresr`)

- **Kept:** `distilbert-base-uncased` and its WordPiece tokenizer, `AutoModelForSequenceClassification` with
  `attn_implementation="eager"`, the strategy grid (frozen backbone / top 2 blocks / full model x three learning rates,
  10 or 6 epochs), AdamW with 10% linear warm-up then linear decay, batch 16, `max_length=24`, per-epoch validation
  scoring with best-epoch selection (validation accuracy, ties: log loss), seed 1234, weights stored in float16.
- **Changed:**
  1. nine classes, 270 test commands;
  2. the primary model is the best validation epoch of the train-only run (the original refit on train + validation is
     `--protocol trainval_refit`);
  3. the pretrained model and tokenizer are pinned to revision `12040accade4e8a0f71eabdb258fecc2e7e948be`
     (`config/distilbert.yaml`);
  4. the float16 weights are always reloaded and evaluated in float32, so reported metrics describe the saved artifact;
  5. the business-value two-way and three-tier gates (tuned with UNKNOWN examples) are replaced by the shared threshold.
     The three-tier idea is re-derived without unknown labels as a secondary analysis (open threshold = default,
     menu threshold = 99% validation coverage, top-3 menu).
- **Not ported:** the masked-language-model illustration, the BERT-base comparison and the cost-sensitivity tables
  (kept in the original notebook).

## Secondary experiments (never mixed into the primary comparison)

| Command | Writes to |
|---|---|
| `python -m scripts.train --model <m> --protocol trainval_refit` then `python -m scripts.evaluate --model <m> --label trainval_refit` | `artifacts/secondary/<m>_trainval_refit/`, `results/secondary/<m>_trainval_refit/` |
| `python -m scripts.train --model transformer --variant no_attention` (or `no_glove`) then `python -m scripts.evaluate --model transformer --label no_attention` | `artifacts/secondary/transformer_no_attention/`, `results/secondary/transformer_no_attention/` |
| every evaluation | `secondary_analyses.json` (temperature scaling, three-tier routing) and `robustness_typos.json` |

`scripts.compare` refuses to build the primary comparison from a folder that contains a non-primary run.
