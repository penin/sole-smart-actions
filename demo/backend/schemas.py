"""Request / response models for the demo API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 2000


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH, description="Customer command")
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Omit to use the configured default")
    model_id: str = "bow"

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class DecideRequest(BaseModel):
    probabilities: dict[str, float]
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_id: str = "bow"


class ApplicationResponse(BaseModel):
    action: str
    model_decision: str
    route_to: str | None
    offers: list[str]
    suggested_intent: str | None


class Decision(BaseModel):
    raw_prediction: str
    confidence: float
    threshold: float
    final_intent: str
    status: Literal["accepted", "fallback_applied"]
    below_threshold: bool
    fallback_applied: bool
    application: ApplicationResponse


class RankedProbability(BaseModel):
    intent: str
    probability: float


class PredictResponse(Decision):
    model_id: str
    model_name: str
    input: str
    class_order: list[str]
    probabilities: dict[str, float]
    ranked: list[RankedProbability]
    default_threshold: float
    recognized_tokens: list[str]
    unknown_tokens: list[str]
    explanation: dict[str, Any]
    inference_ms: float
    workflow: dict[str, Any]


class CompareRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    # One threshold for every model, or omit to use each model's own validation-calibrated default.
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    model_ids: list[str] | None = None

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class CompareEntry(BaseModel):
    model_id: str
    model_name: str
    ok: bool
    error: str | None = None
    prediction: PredictResponse | None = None


class ModelSummary(BaseModel):
    id: str
    name: str
    available: bool
    status: str
    group: str = "primary"
    error: str | None = None
