# Sole Smart Actions — final results (Ricardo: benchmarks + fine-tuned DistilBERT, v5)

Data: `data/Sole_Smart_Actions_Intent_Training_v5_Deduplicated.xlsx`. Thresholds and model choice are made on validation.
Every final model is refit on training + validation before it is scored on test.

## Contents
- `01_sole_intent_classifier_v5.html`: the executed notebook, readable in a browser
- `01_sole_intent_classifier_v5.ipynb`: the notebook to re-run (on Colab with a GPU)
- `figures/`: the deck figures, prefixed with their section. Key ones:
  - `s06_11_three_tier_gate`
  - `s06_11_outcomes_summary`
  - `s06_11_confusion_matrices`
  - `s05_10_confidence_calibrated`
- `tables/`: key tables as CSV
  - final_comparison_test
  - two_way_vs_three_tier_gate
  - cost_sensitivity
  - recall/precision
  - distilbert_grid
  - final_configurations

  `all_intermediate/` holds the rest.
- `model/`: the trained model (DistilBERT, 134 MB) + tokenizer + `routing.json` (thresholds 0.963 / 0.45, temperature 1.075)
- `demo/`: `route()` and an example run

## Takeaways

- **Deploy fine-tuned DistilBERT.** It reaches 99.2% test accuracy (macro F1 0.992). The keyword rule reaches 73.3%, and always predicting the most common intent reaches 12.5%.
- **The benchmarks lose money; DistilBERT makes it.** The keyword rule sends 8.8% of supported test commands into the wrong money workflow and loses about $807k a year. Behind the gate, DistilBERT makes **no wrong routes on supported commands**. That is worth about $139k a year at 5,000 commands a day, and the choice holds for any cost of a money error from $0.50 to $50.
- **Managerial decision: three tiers.**
  - Calibrated confidence **≥ 0.963**: open the workflow.
  - **0.45–0.963**: show a menu with the 3 most likely intents plus "something else".
  - **< 0.45**: "we did not understand".

  On test, 93.3% open directly and the remaining 6.7% get a menu that always contains the right intent. There are no fallbacks and no wrong routes, and value rises to about **$152k a year**.
- **Weak spot: out-of-scope requests.** 6.7% still open a workflow (12% without the menu), mostly `HELP_SUPPORT` and never a money flow.
- **Caveats.**
  - The test set is small (240 + 150 commands).
  - Validation accuracy (0.925) is the conservative figure.
  - The menu and error prices are assumptions for Sole to replace.
