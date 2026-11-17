# Sole Smart Actions — final results (Felipe: Custom Transformer, v5)

Data: `data/Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx`. Same dataset, splits and protocol as Ricardo's v5 notebook (DistilBERT): hyperparameters and thresholds are chosen on validation, and every final model is refit on training + validation before it is scored on test.
Control: Ricardo's two benchmarks give exactly his numbers here (keyword rule 0.733 on test, majority 0.125), so the splits and the protocol are the same.

## Contents
- `01_custom_transformer_v5.html`: the executed notebook, readable in a browser
- `01_custom_transformer_v5.ipynb`: the notebook to re-run (CPU is enough; about 35 minutes, almost all of it the hyperparameter search)
- `figures/`: key figures, prefixed with their section
  - `s04_custom_transformer_validation_curve`
  - `s04_custom_transformer_confusion_matrix`
  - `s05_confidence_raw_vs_calibrated`
- `tables/`: key tables as CSV
  - final_comparison_test (with Ricardo's DistilBERT row, same columns as his table)
  - transformer_grid (the 36 configurations tried)
  - thresholds_validation
  - calibration
  - recall/precision
  - final_configurations
- `model/`: the trained Custom Transformer (Keras weights, 1.7 MB) + vocabulary + `routing.json` (threshold 0.958, temperature 2.302, architecture)
- `demo/`: `route()` and an example run

## Takeaways

- **The Custom Transformer nearly matches DistilBERT on accuracy.** 98.8% test accuracy (macro F1 0.987) vs 99.2% for fine-tuned DistilBERT: one command out of 240. It does it with about 138 thousand parameters, roughly 490 times fewer than DistilBERT's 67 million.
- **Chosen configuration:** 1 Transformer block, 4 attention heads, dropout 0.2, learning rate 1e-3, GloVe embeddings, 27 epochs. It had the highest validation accuracy (0.988) of 36 configurations; all of the top 8 use a learning rate of 1e-3.
- **Behind the calibrated gate it makes money.** With temperature 2.30 and threshold 0.958, it makes **no wrong routes on supported test commands** and is worth about **$123k a year** at 5,000 commands a day (DistilBERT: $139k).
- **Attention and GloVe both matter.** Without attention (averaged embeddings + dense head) test accuracy falls to 95.8%; with random embeddings instead of GloVe it falls to 94.6%.
- **Weak spot: out-of-scope requests.** 15.3% of held-out unsupported commands still open a workflow, and 1.3% (2 of 150) open a money flow; DistilBERT never sends one into a money flow. The ablation without attention rejects far more of them (only 2.0% accepted), at the cost of more fallbacks on supported commands.
- **Recommendation for the team:** deploy DistilBERT as planned. The Custom Transformer is a credible lightweight alternative if model size matters, but its handling of out-of-scope requests is weaker.
- **Caveats.**
  - The test set is small (240 + 150 commands); the gap to DistilBERT is within noise.
  - Calibration (ECE) is measured on the same validation commands used to fit the temperature.
  - Latency was measured on a laptop CPU (38 ms per command); Ricardo's DistilBERT latency comes from a GPU, so the two are not comparable.
  - The error prices are assumptions for Sole to replace.
