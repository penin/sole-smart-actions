"""Model registry: name -> adapter class. Heavy frameworks (torch, keras, transformers) import lazily."""

from __future__ import annotations

import importlib

from .base import IntentModel

_REGISTRY = {
    "bow": "sole_smart_actions.models.bow.model:BowModel",
    "tfidf": "sole_smart_actions.models.tfidf.model:TfidfModel",
    "transformer": "sole_smart_actions.models.transformer.model:CustomTransformerModel",
    "distilbert": "sole_smart_actions.models.distilbert.model:DistilBertModel",
}


def model_class(name: str) -> type[IntentModel]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choose from {sorted(_REGISTRY)}")
    module, cls = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)


def create_model(name: str, cfg: dict) -> IntentModel:
    return model_class(name)(cfg)


__all__ = ["IntentModel", "create_model", "model_class"]
