# Demo: routing function

Model and settings in `../model/` (`routing.json` holds the threshold, the temperature and the architecture). The values come from the notebook run (Sections 4, 5 and 7).
Flow: command → Custom Transformer → probabilities calibrated with the temperature (2.302) → gate:
- confidence **≥ 0.958**: open the predicted workflow
- **< 0.958**: fallback (clarification / help)

## Code (Section 7 of the notebook)
```python
T_TRANSFORMER = TEMPERATURE["Custom Transformer"]
THRESHOLD_TRANSFORMER = float(gate_results.set_index("Model").loc["Custom Transformer", "Threshold"])


def route(command):
    proba = apply_temperature(transformer_predictor(command)[None, :], T_TRANSFORMER)[0]
    top = int(proba.argmax())
    action = f"open {CLASSES[top]}" if proba[top] >= THRESHOLD_TRANSFORMER else "fallback: clarification / help"
    return CLASSES[top], float(proba[top]), action
```

## Output of the run
```
Custom Transformer (T = 2.302, threshold = 0.958)

Send 20 to my mum                   -> TRANSFER_MONEY      99.7%   open TRANSFER_MONEY
transfer didn't go through          -> TRANSFER_MONEY      98.9%   open TRANSFER_MONEY
how much do I have                  -> CHECK_BALANCE       99.7%   open CHECK_BALANCE
show what I withdrew last week      -> VIEW_TRANSACTIONS   99.0%   open VIEW_TRANSACTIONS
pay fiji water                      -> BILL_PAY            71.9%   fallback: clarification / help
who paid into the church group      -> VIEW_GROUP          99.8%   open VIEW_GROUP
book me a taxi                      -> MANAGE_CONTACTS     93.7%   fallback: clarification / help
what's the weather tomorrow         -> VIEW_GROUP          61.7%   fallback: clarification / help
```

These are the same eight commands as Ricardo's DistilBERT demo. One is worth flagging: *"transfer didn't go through"* is a support request (HELP_SUPPORT), but the Custom Transformer opens TRANSFER_MONEY with 98.9% confidence. DistilBERT sends it to a menu with HELP_SUPPORT first.

**Model included** in `../model/`: Keras weights (`model.weights.h5`), the vocabulary (`vocabulary.json`, learned on training + validation) and `routing.json`.
To load it, rebuild the architecture with `build_transformer` from the notebook (the values are in `routing.json`), call `model.load_weights("model/model.weights.h5")`, and turn text into token ids with `TextVectorization(output_sequence_length=16, vocabulary=vocab[2:])` (the first two entries of `vocabulary.json` are padding and `[UNK]`, which the layer adds itself). Divide the log-probabilities by the temperature before applying the threshold.
Checked on this machine: loading the files this way gives the same intents and confidences as the output above.
