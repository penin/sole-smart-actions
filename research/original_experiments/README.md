# Original teammate experiments (preserved for reference)

These folders are the teammates' original deliverables, copied from their ZIP archives so their work, results and
reasoning stay available alongside the integrated project. **They are historical records.** Their numbers come from
each teammate's own protocol and are **not** part of the primary four-model comparison in `results/comparison/`.
The integrated, protocol-aligned implementations live in `src/sole_smart_actions/models/`.

| Folder | Model | Original author | Archive (SHA-256) |
|---|---|---|---|
| `tfidf/` | TF-IDF + Logistic Regression (word, word + char), keyword-rule benchmark, typo robustness, exploratory UNKNOWN training | Chelsea Cheng | `sole-smart-actions-intent.zip` (`68099a7994b6d45c...`) |
| `custom_transformer/` | Custom Transformer from scratch (GloVe, 36-configuration grid, no-attention / no-GloVe ablations, temperature calibration, value-based gate) | Felipe (`friosf`) | `Custom_Transformer.zip` (`bc225fa40c78b045...`) |
| `distilbert/` | Fine-tuned DistilBERT (3 strategies x 3 learning rates), majority and keyword benchmarks, temperature calibration, two-way and three-tier gates, cost sensitivity | Ricardo (`ricardomanjarresr`) | `DistilBERT.zip` (`0d3470ed81e847d2...`) |

Full SHA-256 of the archives:

```
68099a7994b6d45c532fcb1d256cd5cb9141a786526a679a073e2b18f5498a9f  sole-smart-actions-intent.zip
bc225fa40c78b04561571ded870a878a32286cfc626a6b0cb31873cbc47261fa  Custom_Transformer.zip
0d3470ed81e847d2ea08f474c33a83eaf5739e1fd860d97acbcc75219e91b221  DistilBERT.zip
```

## What was changed when copying

- **Not copied:** each archive's copy of the V5 workbook (identical content to `data/raw/`; the TF-IDF copy was
  re-saved and has a different file hash but identical cells), macOS metadata (`__MACOSX/`, `.DS_Store`), and the
  original 8-class DistilBERT weights `distilbert/model/model.safetensors` (134 MB, SHA-256
  `0d51548dc7be7e429b6a51bb53aae1f93cb7eaa5d9dc49e2e49aaa12f11068fa`), which the protocol-aligned 9-class model in
  `artifacts/distilbert/` supersedes. The original file remains in `DistilBERT.zip`.
- **Personal paths removed:** six output lines in the executed notebooks (`.ipynb` / `.html`) showed the teammates'
  local or Colab folders; they now read `<local path>`. Nothing else in the notebooks was edited.

## Why these results differ from the primary comparison

| | Original protocol | Primary protocol (this repository) |
|---|---|---|
| Classes | Custom Transformer and DistilBERT: 8 (CHECK_SPENDING excluded); TF-IDF: 9 | 9 for every model |
| Test set | 240 commands (Transformer, DistilBERT); 270 (TF-IDF) | 270 commands |
| Final model | Transformer, DistilBERT: refit on train + validation | trained on train only; train + validation refit kept as a labelled secondary experiment (`--protocol trainval_refit`) |
| Threshold | value-based / three-tier gates tuned with 150 UNKNOWN examples (neural); 95% validation coverage via `np.quantile` (TF-IDF) | 95% validation known-intent coverage, no unknown labels, identical rule for every model |
| Selection between TF-IDF representations | validation macro F1, then 5-fold CV on train + validation | validation macro F1 (ties: grid order) |

See `docs/model_integration.md` for exactly how each implementation was ported.
