"""
FastAPI Application Entry Point
=================================
Creates, configures, and returns the FastAPI ``app`` instance.

Startup behaviour
-----------------
On startup, the application attempts to load the best trained model from
``artifacts/models/``. If no trained model is found (e.g. first run), the
``/predict`` endpoint will return HTTP 503 until ``POST /train`` is called.

Running the server
------------------
From the project root::

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Or via the convenience script::

    python run.py
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.router import initialise_inference_pipeline, router
from src.logger import get_logger
from src.utils.common import load_config

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown hooks)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load model artefacts on startup; clean up on shutdown."""
    logger.info("━━━ AI Math Olympiad Solver API — Starting up ━━━")
    initialise_inference_pipeline()
    yield
    logger.info("━━━ AI Math Olympiad Solver API — Shutting down ━━━")


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    cfg = load_config("config/config.yaml")
    api_cfg = cfg.get("api", {})

    application = FastAPI(
        title=api_cfg.get("title", "AI Math Olympiad Solver API"),
        version=api_cfg.get("version", "1.0.0"),
        description=api_cfg.get(
            "description",
            "Production-ready REST API for classifying and solving "
            "Math Olympiad problems using ML + Deep Learning models.",
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    application.include_router(router, prefix="/api/v1")

    # ── Root redirect ─────────────────────────────────────────────────────────
    @application.get("/", include_in_schema=False)
    def root():
        return JSONResponse(
            content={
                "message": "AI Math Olympiad Solver API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/api/v1/health",
            }
        )

    return application


app = create_app()
