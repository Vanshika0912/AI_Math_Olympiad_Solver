"""
Pydantic Request / Response Schemas
=====================================
All FastAPI endpoint models live here so that the router stays clean and
validation logic is centralised and independently testable.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Payload accepted by the ``/predict`` endpoint."""

    problem: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        example="Find all prime numbers p such that p^2 + 2 is also prime.",
        description="A math olympiad problem statement.",
    )

    @field_validator("problem")
    @classmethod
    def problem_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Problem text must not be blank or whitespace only.")
        return v.strip()


class TrainRequest(BaseModel):
    """Optional payload accepted by the ``/train`` endpoint."""

    config_path: Optional[str] = Field(
        default="config/config.yaml",
        description="Path to a custom YAML config file (optional).",
    )
    force: bool = Field(
        default=False,
        description="Force re-training even if trained models already exist.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Response returned by the ``/predict`` endpoint."""

    problem: str = Field(..., description="Original input problem text.")
    predicted_category: str = Field(
        ..., description="Predicted Math Olympiad topic category."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's confidence score (0–1)."
    )
    all_probabilities: Dict[str, float] = Field(
        ..., description="Probability for every category."
    )
    solution_approach: str = Field(
        ..., description="High-level methodological hint for the problem."
    )
    model_used: str = Field(
        ..., description="Which model produced the prediction."
    )


class TrainResponse(BaseModel):
    """Response returned by the ``/train`` endpoint."""

    status: str = Field(..., description="'success' or 'error'.")
    message: str = Field(..., description="Human-readable outcome message.")
    metrics: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Evaluation metrics for each model after training.",
    )


class HealthResponse(BaseModel):
    """Response returned by the ``/health`` endpoint."""

    status: str = Field(default="ok", description="Service health status.")
    model_loaded: bool = Field(
        ..., description="Whether the inference pipeline is initialised."
    )
    model_type: Optional[str] = Field(
        default=None, description="Active model type."
    )
    version: str = Field(default="1.0.0", description="API version.")


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    status: str = Field(default="error")
    detail: str = Field(..., description="Error description.")
