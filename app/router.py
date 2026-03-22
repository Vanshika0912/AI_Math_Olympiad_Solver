"""
API Router
===========
Defines and registers all route handlers:

- GET  /health   — liveness / readiness probe
- POST /predict  — classify a math problem
- POST /train    — trigger a full re-training run
"""

import sys
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from src.exception import InferenceError, MathSolverException
from src.logger import get_logger
from src.pipeline.inference_pipeline import InferencePipeline
from src.pipeline.training_pipeline import TrainingPipeline

logger = get_logger(__name__)
router = APIRouter()

# ── Module-level inference pipeline (loaded once at startup) ─────────────────
_inference_pipeline: Optional[InferencePipeline] = None


def get_inference_pipeline() -> InferencePipeline:
    """Return the cached InferencePipeline, raising 503 if not ready."""
    global _inference_pipeline
    if _inference_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not yet trained. "
                "Please call POST /train first to train and save the model."
            ),
        )
    return _inference_pipeline


def initialise_inference_pipeline(config_path: str = "config/config.yaml") -> None:
    """Attempt to load the saved model. Silently skip if artefacts are absent."""
    global _inference_pipeline
    try:
        _inference_pipeline = InferencePipeline(config_path)
        logger.info("Inference pipeline initialised at startup.")
    except Exception as exc:
        logger.warning(
            "Could not load model at startup (train first): %s", exc
        )
        _inference_pipeline = None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Monitoring"],
)
def health_check() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns whether the service is running and whether the inference model
    has been loaded successfully.
    """
    loaded = _inference_pipeline is not None
    model_type = (
        _inference_pipeline._model_type if loaded else None
    )
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_type=model_type,
        version="1.0.0",
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Math Problem Category",
    tags=["Inference"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not trained yet"},
    },
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Classify a math olympiad problem.

    Accepts a problem statement and returns:
    - Predicted topic category (e.g. *Number Theory*, *Geometry*)
    - Confidence score
    - Per-class probabilities
    - A concise solution approach hint
    """
    pipeline = get_inference_pipeline()
    try:
        result = pipeline.predict(request.problem)
        return PredictResponse(
            problem=request.problem,
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            solution_approach=result["solution_approach"],
            model_used=result["model_used"],
        )
    except InferenceError as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc}",
        )


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Trigger Model Re-Training",
    tags=["Training"],
    responses={
        500: {"model": ErrorResponse, "description": "Training failed"},
    },
)
def train(request: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
    """
    Kick off a full training pipeline run.

    The pipeline executes **synchronously** so the caller receives the
    training result (including metrics) in the HTTP response.
    Re-training replaces all model artefacts in ``artifacts/models/``.
    """
    global _inference_pipeline
    try:
        logger.info("Re-training requested via API (force=%s).", request.force)

        pipeline = TrainingPipeline(config_path=request.config_path)
        pipeline.run()

        # Reload the inference pipeline from the freshly trained model
        _inference_pipeline = InferencePipeline(config_path=request.config_path)
        logger.info("Inference pipeline reloaded after training.")

        return TrainResponse(
            status="success",
            message="Training completed successfully. Inference pipeline reloaded.",
            metrics=None,
        )

    except MathSolverException as exc:
        logger.error("Training pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during training.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc}",
        )
