T_TRANSFORMER = TEMPERATURE["Custom Transformer"]
THRESHOLD_TRANSFORMER = float(gate_results.set_index("Model").loc["Custom Transformer", "Threshold"])


def route(command):
    proba = apply_temperature(transformer_predictor(command)[None, :], T_TRANSFORMER)[0]
    top = int(proba.argmax())
    action = f"open {CLASSES[top]}" if proba[top] >= THRESHOLD_TRANSFORMER else "fallback: clarification / help"
    return CLASSES[top], float(proba[top]), action


print(f"Custom Transformer (T = {T_TRANSFORMER:.3f}, threshold = {THRESHOLD_TRANSFORMER:.3f})\n")
for command in ["Send 20 to my mum", "transfer didn't go through", "how much do I have",
                "show what I withdrew last week", "pay fiji water", "who paid into the church group",
                "book me a taxi", "what's the weather tomorrow"]:
    intent, confidence, action = route(command)
    print(f"{command:35s} -> {intent:18s} {confidence:6.1%}   {action}")