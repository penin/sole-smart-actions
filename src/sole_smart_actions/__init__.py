"""Sole Smart Actions: natural-language intent classification for the Sole digital wallet (Fiji).

One canonical dataset (V5), one shared evaluation protocol and four interchangeable models:
Bag of Words, TF-IDF, a Custom Transformer and fine-tuned DistilBERT. The system only
classifies intents and simulates navigation; it never executes financial transactions.
"""

__version__ = "2.0.0"

MODEL_NAMES = ("bow", "tfidf", "transformer", "distilbert")
