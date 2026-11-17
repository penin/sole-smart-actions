# Demo: routing function

Model and settings in `../model/` (`routing.json` holds the thresholds and temperature). The values come from the notebook run (Sections 5 and 6.9).
Flow: command → DistilBERT → probabilities calibrated with the temperature (1.075) → three-tier gate:
- confidence **≥ 0.963**: open the predicted workflow
- **0.45–0.963**: menu with the 3 most likely intents + "something else"
- **< 0.45**: "we did not understand": ask to rephrase / help

## Code (Section 7 of the notebook)
```python
NOT_UNDERSTOOD = "not understood: ask to rephrase / help"


def route(command, name=None):
    """Intent, calibrated confidence and the action the app would take for one command."""
    name = name or DEPLOYED_NAME
    if name == DEPLOYED_NAME:
        open_threshold, menu_threshold = OPEN_THRESHOLD, MENU_THRESHOLD
    else:   # other models keep their two-way gate from Section 5
        open_threshold = menu_threshold = float(gate_results.set_index("Model").loc[name, "Threshold"])
    proba = apply_temperature(PREDICTORS[name](command)[None, :], TEMPERATURE[name])
    pred, confidence = int(predict_labels(proba)[0]), float(proba.max())
    if pred == ABSTAIN:
        return "—", confidence, NOT_UNDERSTOOD
    intent = CLASSES[pred]
    if confidence >= open_threshold:
        action = f"open {intent}"
    elif confidence >= menu_threshold:
        options = [CLASSES[k] for k in np.argsort(-proba[0])[:MENU_SIZE]]
        action = "menu: " + " / ".join(options) + " / something else"
    else:
        action = NOT_UNDERSTOOD
    return intent, confidence, action


print(f"Routing with {DEPLOYED_NAME} (open >= {OPEN_THRESHOLD:.3f}, menu >= {MENU_THRESHOLD:.2f})\n")
for command in ["Send 20 to my mum", "transfer didn't go through", "how much do I have",
                "show what I withdrew last week", "pay fiji water", "who paid into the church group",
                "book me a taxi", "what's the weather tomorrow"]:
    intent, confidence, action = route(command)
    print(f"{command:35s} -> {intent:18s} {confidence:6.1%}   {action}")
```

## Output of the run
```
Routing with Fine-tuned DistilBERT (open >= 0.963, menu >= 0.45)

Send 20 to my mum                   -> TRANSFER_MONEY      98.9%   open TRANSFER_MONEY
transfer didn't go through          -> HELP_SUPPORT        68.8%   menu: HELP_SUPPORT / VIEW_TRANSACTIONS / MANAGE_CONTACTS / something else
how much do I have                  -> CHECK_BALANCE       98.3%   open CHECK_BALANCE
show what I withdrew last week      -> VIEW_TRANSACTIONS   73.1%   menu: VIEW_TRANSACTIONS / HELP_SUPPORT / CHECK_BALANCE / something else
pay fiji water                      -> BILL_PAY            98.6%   open BILL_PAY
who paid into the church group      -> VIEW_GROUP          98.6%   open VIEW_GROUP
book me a taxi                      -> BILL_PAY            50.8%   menu: BILL_PAY / WITHDRAW_MONEY / HELP_SUPPORT / something else
what's the weather tomorrow         -> CHECK_BALANCE       80.8%   menu: CHECK_BALANCE / VIEW_TRANSACTIONS / HELP_SUPPORT / something else
```

**Model included** in `../model/`: fine-tuned DistilBERT in half precision (134 MB), with its tokenizer and `routing.json`.
Load it with `AutoModelForSequenceClassification.from_pretrained("model")` and `AutoTokenizer.from_pretrained("model")`.
It was checked on this machine: it gives the same intents and confidences as the output above.
