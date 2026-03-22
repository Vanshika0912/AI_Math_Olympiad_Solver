"""
Inference Pipeline
===================
Loads the saved preprocessor and best model, then classifies a raw math
problem string without any re-training.

The pipeline is intentionally stateless between calls: artefacts are loaded
once at construction time and reused for every subsequent ``predict()`` call,
making it safe to use inside long-running API processes.

Returned prediction schema
--------------------------
{
    "predicted_category":   str,          # e.g. "Number Theory"
    "confidence":           float,        # probability of top class
    "all_probabilities":    dict[str, float],  # every class's probability
    "model_used":           str,          # "random_forest" | "deep_learning"
    "solution_approach":    str,          # short methodological hint
}
"""

import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.exception import InferenceError
from src.logger import get_logger
from src.models.neural_network import MathProblemClassifier
from src.utils.common import load_config, load_object

logger = get_logger(__name__)

# ── Per-category solution approach hints ────────────────────────────────────
_APPROACH_HINTS: Dict[str, str] = {
    "Algebra": (
        "Look for algebraic identities, substitution, or factorisation. "
        "Isolate variables and simplify using standard polynomial techniques."
    ),
    "Calculus": (
        "Apply differentiation or integration rules. "
        "Check for chain rule, product rule, or fundamental theorem applications."
    ),
    "Combinatorics": (
        "Use counting principles: permutations, combinations, or inclusion-exclusion. "
        "Identify whether order matters."
    ),
    "Geometry": (
        "Draw a diagram. Apply angle theorems, congruence/similarity rules, "
        "or coordinate geometry as appropriate."
    ),
    "Number Theory": (
        "Consider modular arithmetic, prime factorisation, or divisibility rules. "
        "Look for patterns in remainders."
    ),
    "Probability": (
        "Define the sample space clearly. Apply basic probability rules, "
        "conditional probability, or Bayes' theorem."
    ),
    "Series & Sequences": (
        "Identify whether the series is arithmetic, geometric, or telescoping. "
        "Apply summation formulas or convergence tests."
    ),
}


class InferencePipeline:
    """
    Single-instance inference engine for Math Olympiad problem classification.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        try:
            cfg = load_config(config_path)
            self._models_dir = Path(cfg["paths"]["models_dir"])
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            logger.info("Loading inference artefacts …")
            self._preprocessor = load_object(
                self._models_dir / "preprocessor.joblib"
            )
            self._best_model_bundle = load_object(
                self._models_dir / "best_model.joblib"
            )
            self._model_type: str = self._best_model_bundle["type"]
            logger.info("Active model type: %s", self._model_type)

            if self._model_type == "deep_learning":
                self._dl_model = self._load_dl_model()

            logger.info("InferencePipeline ready.")
        except Exception as exc:
            raise InferenceError(str(exc), sys) from exc

    # ── public API ───────────────────────────────────────────────────────────

    def predict(self, problem_text: str) -> Dict:
        """
        Classify a math problem and return a structured result.

        Parameters
        ----------
        problem_text : str
            Raw math problem string (as submitted by the user).

        Returns
        -------
        dict
        """
        try:
            if not problem_text or not problem_text.strip():
                raise InferenceError("Input problem text cannot be empty.")

            clean_text = self._clean(problem_text)
            le = self._preprocessor["label_encoder"]

            if self._model_type == "random_forest":
                result = self._predict_rf(clean_text, le)
            else:
                result = self._predict_dl(clean_text, le)

            # Attach solution approach hint
            category = result["predicted_category"]
            result["solution_approach"] = _APPROACH_HINTS.get(
                category,
                "Apply systematic mathematical reasoning and check for known patterns.",
            )
            result["model_used"] = self._model_type
            return result

        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(str(exc), sys) from exc

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Apply the same cleaning rules used during preprocessing."""
        text = text.lower()
        text = re.sub(
            r"[^a-z0-9\s\+\-\*\/\^\=\(\)\.\,\<\>\≤\≥\≡\∞\π]", " ", text
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _predict_rf(self, clean_text: str, le) -> Dict:
        """Classify using the Random Forest model."""
        tfidf = self._preprocessor["tfidf"]
        rf_model = self._best_model_bundle["model"]

        vec = tfidf.transform([clean_text]).toarray()
        predicted_idx = int(rf_model.predict(vec)[0])
        predicted_category = le.inverse_transform([predicted_idx])[0]

        if hasattr(rf_model, "predict_proba"):
            proba = rf_model.predict_proba(vec)[0]
            confidence = float(proba[predicted_idx])
            all_proba = {
                str(le.inverse_transform([i])[0]): float(p)
                for i, p in enumerate(proba)
            }
        else:
            confidence = 1.0
            all_proba = {str(predicted_category): 1.0}

        return {
            "predicted_category": str(predicted_category),
            "confidence": round(confidence, 4),
            "all_probabilities": all_proba,
        }

    def _predict_dl(self, clean_text: str, le) -> Dict:
        """Classify using the Deep Learning model."""
        vocab: Dict[str, int] = self._preprocessor["vocab"]
        max_seq_len: int = self._preprocessor["max_seq_len"]

        tokens = clean_text.split()[:max_seq_len]
        unk_idx = vocab.get("<UNK>", 1)
        seq = [vocab.get(tok, unk_idx) for tok in tokens]
        seq += [0] * (max_seq_len - len(seq))

        tensor = torch.tensor([seq], dtype=torch.long).to(self._device)
        proba_tensor = self._dl_model.predict_proba(tensor)
        proba = proba_tensor.cpu().numpy()[0]

        predicted_idx = int(np.argmax(proba))
        predicted_category = le.inverse_transform([predicted_idx])[0]
        confidence = float(proba[predicted_idx])
        all_proba = {
            str(le.inverse_transform([i])[0]): round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return {
            "predicted_category": str(predicted_category),
            "confidence": round(confidence, 4),
            "all_probabilities": all_proba,
        }

    def _load_dl_model(self) -> MathProblemClassifier:
        """Reconstruct the PyTorch model from saved state_dict."""
        meta = self._best_model_bundle["metadata"]
        model = MathProblemClassifier(
            vocab_size=meta["vocab_size"],
            embedding_dim=meta["embedding_dim"],
            hidden_dims=meta["hidden_dims"],
            num_classes=meta["num_classes"],
            dropout=meta["dropout"],
        ).to(self._device)
        model.load_state_dict(self._best_model_bundle["state_dict"])
        model.eval()
        logger.info(
            "DL model loaded (vocab=%d, classes=%d).",
            meta["vocab_size"],
            meta["num_classes"],
        )
        return model
