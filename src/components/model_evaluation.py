"""
Model Evaluation Component
===========================
Evaluates both the classical ML model and the deep-learning model on the
held-out test split.

Metrics computed
----------------
- Accuracy
- Precision (weighted)
- Recall    (weighted)
- F1 Score  (weighted)

Artefacts produced
------------------
- ``artifacts/plots/confusion_matrix_<model>.png``
- ``artifacts/plots/metrics_comparison.png``
- ``artifacts/models/best_model.joblib``   ← whichever model wins on F1
- ``artifacts/models/evaluation_report.joblib``
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.components.data_preprocessing import PreprocessedData
from src.components.model_trainer import TrainedModels
from src.exception import ModelEvaluationError
from src.logger import get_logger
from src.models.neural_network import MathProblemClassifier, MathProblemDataset
from src.utils.common import ensure_directories, format_metrics, load_config, save_object

logger = get_logger(__name__)


class ModelEvaluation:
    """
    Computes evaluation metrics for both models, generates comparison plots,
    and saves the best model to disk.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        cfg = load_config(config_path)
        self.plots_dir = Path(cfg["paths"]["plots_dir"])
        self.models_dir = Path(cfg["paths"]["models_dir"])
        self.batch_size = cfg["deep_learning"]["batch_size"]
        self.average = cfg["evaluation"]["average"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── public API ───────────────────────────────────────────────────────────

    def initiate(
        self,
        data: PreprocessedData,
        trained_models: TrainedModels,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both models and return a metrics dictionary.

        Parameters
        ----------
        data : PreprocessedData
            Preprocessed test features and labels.
        trained_models : TrainedModels
            Container holding the two fitted models.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{"random_forest": {...}, "deep_learning": {...}}``
        """
        try:
            logger.info("── Model Evaluation started ──")
            ensure_directories([self.plots_dir, self.models_dir])

            # ── Random Forest ────────────────────────────────────────────────
            rf_preds = trained_models.rf_model.predict(data.X_test_tfidf)
            rf_metrics = self._compute_metrics(data.y_test, rf_preds)
            logger.info(format_metrics(rf_metrics, "Random Forest"))

            self._plot_confusion_matrix(
                y_true=data.y_test,
                y_pred=rf_preds,
                classes=list(data.label_encoder.classes_),
                model_name="Random Forest",
                save_path=self.plots_dir / "confusion_matrix_random_forest.png",
            )

            # ── Deep Learning ────────────────────────────────────────────────
            dl_preds = self._predict_dl(
                trained_models.dl_model,
                data.X_test_seq,
            )
            dl_metrics = self._compute_metrics(data.y_test, dl_preds)
            logger.info(format_metrics(dl_metrics, "Deep Learning"))

            self._plot_confusion_matrix(
                y_true=data.y_test,
                y_pred=dl_preds,
                classes=list(data.label_encoder.classes_),
                model_name="Deep Learning",
                save_path=self.plots_dir / "confusion_matrix_deep_learning.png",
            )

            # ── Comparison plot ──────────────────────────────────────────────
            all_metrics = {
                "Random Forest": rf_metrics,
                "Deep Learning": dl_metrics,
            }
            self._plot_metrics_comparison(
                metrics_dict=all_metrics,
                save_path=self.plots_dir / "metrics_comparison.png",
            )

            # ── Loss curve for DL ────────────────────────────────────────────
            history = trained_models.dl_metadata.get("history", {})
            if history:
                self._plot_loss_curve(
                    history=history,
                    save_path=self.plots_dir / "dl_loss_curve.png",
                )

            # ── Select & save best model ─────────────────────────────────────
            best_model_name, best_artifact = self._select_best_model(
                rf_metrics=rf_metrics,
                dl_metrics=dl_metrics,
                rf_model=trained_models.rf_model,
                dl_model=trained_models.dl_model,
                dl_metadata=trained_models.dl_metadata,
            )

            report = {
                "best_model": best_model_name,
                "random_forest": rf_metrics,
                "deep_learning": dl_metrics,
                "label_classes": list(data.label_encoder.classes_),
            }
            save_object(self.models_dir / "evaluation_report.joblib", report)

            logger.info("Best model selected: %s", best_model_name)
            logger.info("── Model Evaluation complete ──")

            return all_metrics

        except ModelEvaluationError:
            raise
        except Exception as exc:
            raise ModelEvaluationError(str(exc), sys) from exc

    # ── private helpers ──────────────────────────────────────────────────────

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Return accuracy, precision, recall, and F1 as a flat dict."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average=self.average, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average=self.average, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average=self.average, zero_division=0)
            ),
        }

    def _predict_dl(
        self,
        model: MathProblemClassifier,
        X_seq: List[List[int]],
    ) -> np.ndarray:
        """Run inference with the PyTorch model; return class-index array."""
        dummy_labels = [0] * len(X_seq)
        ds = MathProblemDataset(X_seq, dummy_labels)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        model.eval()
        all_preds: List[int] = []
        with torch.no_grad():
            for sequences, _ in loader:
                sequences = sequences.to(self.device)
                logits = model(sequences)
                preds = logits.argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
        return np.array(all_preds)

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[str],
        model_name: str,
        save_path: Path,
    ) -> None:
        """Save a Seaborn heatmap of the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes) - 1)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", save_path)

    def _plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Path,
    ) -> None:
        """Bar-chart comparing all metrics across models."""
        model_names = list(metrics_dict.keys())
        metric_names = ["accuracy", "precision", "recall", "f1_score"]
        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, name in enumerate(model_names):
            values = [metrics_dict[name][m] for m in metric_names]
            bars = ax.bar(x + i * width, values, width, label=name)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison", fontsize=14, pad=12)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Metrics comparison plot saved → %s", save_path)

    def _plot_loss_curve(
        self,
        history: Dict,
        save_path: Path,
    ) -> None:
        """Plot training and validation loss over epochs."""
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="salmon")
        axes[0].set_title("Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(linestyle="--", alpha=0.5)

        axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="seagreen")
        axes[1].set_title("Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(linestyle="--", alpha=0.5)

        plt.suptitle("Deep Learning Training History", fontsize=13)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss curve saved → %s", save_path)

    def _select_best_model(
        self,
        rf_metrics: Dict[str, float],
        dl_metrics: Dict[str, float],
        rf_model,
        dl_model: MathProblemClassifier,
        dl_metadata: Dict,
    ) -> Tuple[str, object]:
        """Compare F1 scores and save the winner as ``best_model.joblib``."""
        rf_f1 = rf_metrics["f1_score"]
        dl_f1 = dl_metrics["f1_score"]

        best_model_path = self.models_dir / "best_model.joblib"

        if rf_f1 >= dl_f1:
            save_object(best_model_path, {"type": "random_forest", "model": rf_model})
            logger.info(
                "Random Forest selected (F1=%.4f vs DL F1=%.4f)", rf_f1, dl_f1
            )
            return "random_forest", rf_model
        else:
            state_dict = {k: v.cpu() for k, v in dl_model.state_dict().items()}
            save_object(
                best_model_path,
                {"type": "deep_learning", "state_dict": state_dict, "metadata": dl_metadata},
            )
            logger.info(
                "Deep Learning selected (F1=%.4f vs RF F1=%.4f)", dl_f1, rf_f1
            )
            return "deep_learning", dl_model
