"""
Try the trained model on your own commands.

    python src/predict.py "Send $50 to Jone" "Show me my recent transactions" "Book a taxi"

Run src/run_experiments.py first (it saves results/best_model.pkl).
Commands below the confidence threshold are routed to a safe fallback
(clarifying question / Help & Support) instead of opening a workflow.
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = pickle.load(open(ROOT / "results" / "best_model.pkl", "rb"))
cfg = json.load(open(ROOT / "results" / "best_model_config.json"))
threshold = cfg["threshold"]

commands = sys.argv[1:] or ["Send $50 to Jone", "Show me my recent transactions",
                            "Pay my EFL bill", "Snd 20 to litia", "Book a taxi"]
print(f"Model: {cfg['best_model']}  |  confidence threshold: {threshold:.2f}\n")
for text in commands:
    proba = model.predict_proba([text])[0]
    i = proba.argmax()
    intent, conf = model.classes_[i], proba[i]
    action = intent if conf >= threshold else "FALLBACK -> ask user to clarify / HELP_SUPPORT"
    print(f"{text!r:45s} -> {intent:18s} conf={conf:.2f}  route: {action}")
