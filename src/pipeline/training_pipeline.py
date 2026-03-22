"""
Training Pipeline Orchestrator
================================
Chains all pipeline components into a single, reproducible training run:

  DataIngestion → DataPreprocessing → ModelTrainer → ModelEvaluation

Usage (from project root)::

    python -m src.pipeline.training_pipeline
"""

import sys
import time

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.exception import MathSolverException
from src.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """
    End-to-end orchestrator for the training workflow.

    All component classes read from the same ``config/config.yaml`` so a
    single config change propagates throughout the pipeline.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path = config_path

    def run(self) -> None:
        """Execute the full training pipeline end-to-end."""
        try:
            start = time.time()
            logger.info("╔══════════════════════════════════════════╗")
            logger.info("║   AI Math Olympiad Solver — Training     ║")
            logger.info("╚══════════════════════════════════════════╝")

            # ── Step 1: Data Ingestion ────────────────────────────────────────
            logger.info("[1/4] Data Ingestion …")
            ingestion = DataIngestion(self.config_path)
            train_path, test_path = ingestion.initiate()

            # ── Step 2: Data Preprocessing ────────────────────────────────────
            logger.info("[2/4] Data Preprocessing …")
            preprocessing = DataPreprocessing(self.config_path)
            preprocessed_data = preprocessing.initiate()

            # ── Step 3: Model Training ────────────────────────────────────────
            logger.info("[3/4] Model Training …")
            trainer = ModelTrainer(self.config_path)
            trained_models = trainer.initiate(preprocessed_data)

            # ── Step 4: Model Evaluation ──────────────────────────────────────
            logger.info("[4/4] Model Evaluation …")
            evaluator = ModelEvaluation(self.config_path)
            metrics = evaluator.initiate(preprocessed_data, trained_models)

            elapsed = time.time() - start
            logger.info("━" * 48)
            logger.info("Training pipeline completed in %.1f s.", elapsed)
            logger.info("Metrics summary:")
            for model_name, model_metrics in metrics.items():
                for metric, value in model_metrics.items():
                    logger.info("  %-20s %-15s %.4f", model_name, metric, value)
            logger.info("━" * 48)

        except MathSolverException:
            raise
        except Exception as exc:
            raise MathSolverException(str(exc), sys) from exc


if __name__ == "__main__":
    TrainingPipeline().run()
