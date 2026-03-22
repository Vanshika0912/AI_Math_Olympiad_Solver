"""
Model Trainer Component
========================
Trains two models on preprocessed Math Olympiad data:

1. ``RandomForestClassifier`` (scikit-learn) — classical ML baseline.
2. ``MathProblemClassifier``  (PyTorch)       — embedding-based deep learner.

Both models are persisted to ``artifacts/models/`` after training.
The trainer also supports early-stopping for the neural network to avoid
over-fitting on smaller datasets.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.components.data_preprocessing import PreprocessedData
from src.exception import ModelTrainingError
from src.logger import get_logger
from src.models.neural_network import MathProblemClassifier, MathProblemDataset
from src.utils.common import load_config, save_object

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Typed container for training hyper-parameters."""

    # Paths
    rf_model_path: str
    dl_model_path: str
    dl_metadata_path: str

    # Random Forest
    rf_n_estimators: int
    rf_max_depth: int
    rf_min_samples_split: int
    rf_min_samples_leaf: int
    rf_random_state: int

    # Deep Learning
    embedding_dim: int
    hidden_dims: List[int]
    dropout: float
    learning_rate: float
    epochs: int
    batch_size: int
    patience: int
    weight_decay: float


@dataclass
class TrainedModels:
    """Container returned by the trainer holding both trained models."""

    rf_model: RandomForestClassifier
    dl_model: MathProblemClassifier
    dl_metadata: Dict


class ModelTrainer:
    """
    Orchestrates training of the classical and deep-learning models.

    Parameters
    ----------
    config_path : str
        Path to the central YAML configuration file.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        cfg = load_config(config_path)
        dl_cfg = cfg["deep_learning"]
        rf_cfg = cfg["traditional_model"]["params"]

        self.config = TrainerConfig(
            rf_model_path="artifacts/models/random_forest_model.joblib",
            dl_model_path="artifacts/models/deep_learning_model.pt",
            dl_metadata_path="artifacts/models/dl_metadata.joblib",
            rf_n_estimators=rf_cfg["n_estimators"],
            rf_max_depth=rf_cfg["max_depth"],
            rf_min_samples_split=rf_cfg["min_samples_split"],
            rf_min_samples_leaf=rf_cfg["min_samples_leaf"],
            rf_random_state=rf_cfg["random_state"],
            embedding_dim=dl_cfg["embedding_dim"],
            hidden_dims=dl_cfg["hidden_dims"],
            dropout=dl_cfg["dropout"],
            learning_rate=dl_cfg["learning_rate"],
            epochs=dl_cfg["epochs"],
            batch_size=dl_cfg["batch_size"],
            patience=dl_cfg["early_stopping_patience"],
            weight_decay=dl_cfg["weight_decay"],
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training device: %s", self.device)

    # ── public API ───────────────────────────────────────────────────────────

    def initiate(self, data: PreprocessedData) -> TrainedModels:
        """
        Train both models and persist them to disk.

        Parameters
        ----------
        data : PreprocessedData
            Output of the preprocessing pipeline.

        Returns
        -------
        TrainedModels
        """
        try:
            logger.info("── Model Training started ──")

            rf_model = self._train_random_forest(
                data.X_train_tfidf, data.y_train
            )
            dl_model, dl_metadata = self._train_deep_learning(
                data.X_train_seq,
                data.X_test_seq,
                data.y_train,
                data.y_test,
                vocab_size=len(data.vocab),
                num_classes=data.num_classes,
            )

            self._save_models(rf_model, dl_model, dl_metadata)
            logger.info("── Model Training complete ──")

            return TrainedModels(
                rf_model=rf_model,
                dl_model=dl_model,
                dl_metadata=dl_metadata,
            )

        except ModelTrainingError:
            raise
        except Exception as exc:
            raise ModelTrainingError(str(exc), sys) from exc

    # ── private helpers ──────────────────────────────────────────────────────

    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> RandomForestClassifier:
        """Fit and return a RandomForestClassifier."""
        logger.info("Training RandomForestClassifier …")
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.rf_random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        logger.info(
            "RandomForest trained in %.1f s  |  %d trees",
            time.time() - t0,
            self.config.rf_n_estimators,
        )
        return rf

    def _train_deep_learning(
        self,
        X_train_seq: List[List[int]],
        X_val_seq: List[List[int]],
        y_train: np.ndarray,
        y_val: np.ndarray,
        vocab_size: int,
        num_classes: int,
    ) -> Tuple[MathProblemClassifier, Dict]:
        """Fit and return a MathProblemClassifier with early stopping."""
        logger.info("Training MathProblemClassifier (PyTorch) …")

        train_ds = MathProblemDataset(X_train_seq, y_train.tolist())
        val_ds = MathProblemDataset(X_val_seq, y_val.tolist())

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        model = MathProblemClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
            num_classes=num_classes,
            dropout=self.config.dropout,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=False
        )

        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        patience_counter = 0
        history: Dict = {"train_loss": [], "val_loss": [], "val_acc": []}

        t0 = time.time()
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            epoch_loss = self._run_epoch(model, train_loader, criterion, optimizer)

            val_loss, val_acc = self._evaluate_epoch(model, val_loader, criterion)
            scheduler.step(val_loss)

            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch,
                self.config.epochs,
                epoch_loss,
                val_loss,
                val_acc,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d (patience=%d).",
                        epoch,
                        self.config.patience,
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info("Best weights restored (val_loss=%.4f).", best_val_loss)

        logger.info(
            "Deep-learning training complete in %.1f s.", time.time() - t0
        )

        dl_metadata = {
            "vocab_size": vocab_size,
            "embedding_dim": self.config.embedding_dim,
            "hidden_dims": self.config.hidden_dims,
            "num_classes": num_classes,
            "dropout": self.config.dropout,
            "history": history,
        }

        return model, dl_metadata

    def _run_epoch(
        self,
        model: MathProblemClassifier,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch; return mean loss."""
        total_loss = 0.0
        for sequences, labels in loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        return total_loss / len(loader.dataset)

    def _evaluate_epoch(
        self,
        model: MathProblemClassifier,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Run one validation pass; return (mean loss, accuracy)."""
        model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for sequences, labels in loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                logits = model(sequences)
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
        n = len(loader.dataset)
        return total_loss / n, correct / n

    def _save_models(
        self,
        rf_model: RandomForestClassifier,
        dl_model: MathProblemClassifier,
        dl_metadata: Dict,
    ) -> None:
        """Persist both models to the artifact store."""
        Path(self.config.rf_model_path).parent.mkdir(parents=True, exist_ok=True)

        save_object(self.config.rf_model_path, rf_model)
        logger.info("RandomForest saved → %s", self.config.rf_model_path)

        torch.save(dl_model.state_dict(), self.config.dl_model_path)
        logger.info("DL model weights saved → %s", self.config.dl_model_path)

        save_object(self.config.dl_metadata_path, dl_metadata)
        logger.info("DL metadata saved → %s", self.config.dl_metadata_path)
