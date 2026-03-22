"""
Data Preprocessing Component
=============================
Responsibilities
----------------
1. Load train / test CSV artefacts produced by DataIngestion.
2. Clean and normalise problem text (lower-case, punctuation removal, etc.).
3. Encode labels with ``LabelEncoder``.
4. Build a TF-IDF feature matrix for the classical ML branch.
5. Build vocabulary-indexed integer sequences for the PyTorch neural net.
6. Persist the fitted ``preprocessor`` bundle to disk so the inference
   pipeline can replicate transformations without re-fitting.

Output artefacts
----------------
- ``artifacts/models/preprocessor.joblib``  — fitted sklearn pipeline + vocab
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from src.exception import DataPreprocessingError
from src.logger import get_logger
from src.utils.common import ensure_directories, load_config, save_object

logger = get_logger(__name__)


@dataclass
class PreprocessorConfig:
    """Typed container for preprocessing settings."""

    text_column: str
    target_column: str
    train_data_path: str
    test_data_path: str
    preprocessor_path: str
    processed_data_path: str
    max_features: int
    ngram_range: Tuple[int, int]
    min_df: int
    max_df: float
    sublinear_tf: bool


@dataclass
class PreprocessedData:
    """Container returned by the preprocessing pipeline."""

    X_train_tfidf: np.ndarray
    X_test_tfidf: np.ndarray
    X_train_seq: List[List[int]]
    X_test_seq: List[List[int]]
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    vocab: Dict[str, int]
    max_seq_len: int
    num_classes: int


class DataPreprocessing:
    """
    Orchestrates text cleaning, feature engineering, and label encoding.

    Parameters
    ----------
    config_path : str | Path
        Path to the central YAML configuration file.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        cfg = load_config(config_path)
        ngram = cfg["preprocessing"]["ngram_range"]
        self.config = PreprocessorConfig(
            text_column=cfg["data"]["text_column"],
            target_column=cfg["data"]["target_column"],
            train_data_path=cfg["paths"]["train_data_path"],
            test_data_path=cfg["paths"]["test_data_path"],
            preprocessor_path="artifacts/models/preprocessor.joblib",
            processed_data_path=cfg["paths"]["processed_data_path"],
            max_features=cfg["preprocessing"]["max_features"],
            ngram_range=tuple(ngram),
            min_df=cfg["preprocessing"]["min_df"],
            max_df=cfg["preprocessing"]["max_df"],
            sublinear_tf=cfg["preprocessing"]["sublinear_tf"],
        )

    # ── public API ───────────────────────────────────────────────────────────

    def initiate(self) -> PreprocessedData:
        """
        Execute the full preprocessing pipeline.

        Returns
        -------
        PreprocessedData
        """
        try:
            logger.info("── Data Preprocessing started ──")

            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            train_texts = train_df[self.config.text_column].tolist()
            test_texts = test_df[self.config.text_column].tolist()

            # ── Text cleaning ────────────────────────────────────────────────
            train_clean = [self._clean(t) for t in train_texts]
            test_clean = [self._clean(t) for t in test_texts]

            # ── Label encoding ───────────────────────────────────────────────
            le = LabelEncoder()
            y_train = le.fit_transform(train_df[self.config.target_column])
            y_test = le.transform(test_df[self.config.target_column])
            num_classes = len(le.classes_)
            logger.info("Classes: %s", list(le.classes_))

            # ── TF-IDF features ──────────────────────────────────────────────
            tfidf = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                sublinear_tf=self.config.sublinear_tf,
            )
            X_train_tfidf = tfidf.fit_transform(train_clean).toarray()
            X_test_tfidf = tfidf.transform(test_clean).toarray()
            logger.info("TF-IDF feature matrix: %s", X_train_tfidf.shape)

            # ── Integer-sequence features (for PyTorch) ──────────────────────
            vocab, max_seq_len = self._build_vocab(train_clean)
            X_train_seq = self._texts_to_sequences(train_clean, vocab, max_seq_len)
            X_test_seq = self._texts_to_sequences(test_clean, vocab, max_seq_len)
            logger.info(
                "Vocab size: %d | Max seq len: %d", len(vocab), max_seq_len
            )

            # ── Persist preprocessor bundle ──────────────────────────────────
            preprocessor_bundle = {
                "tfidf": tfidf,
                "label_encoder": le,
                "vocab": vocab,
                "max_seq_len": max_seq_len,
            }
            ensure_directories([Path(self.config.preprocessor_path).parent])
            save_object(self.config.preprocessor_path, preprocessor_bundle)

            logger.info("Preprocessor saved to %s", self.config.preprocessor_path)
            logger.info("── Data Preprocessing complete ──")

            return PreprocessedData(
                X_train_tfidf=X_train_tfidf,
                X_test_tfidf=X_test_tfidf,
                X_train_seq=X_train_seq,
                X_test_seq=X_test_seq,
                y_train=y_train,
                y_test=y_test,
                label_encoder=le,
                vocab=vocab,
                max_seq_len=max_seq_len,
                num_classes=num_classes,
            )

        except DataPreprocessingError:
            raise
        except Exception as exc:
            raise DataPreprocessingError(str(exc), sys) from exc

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """
        Normalise a single math problem string.

        Steps
        -----
        - Lower-case
        - Keep alphanumeric characters, basic math symbols, and spaces
        - Collapse whitespace
        """
        text = text.lower()
        # Keep letters, digits, and key math tokens
        text = re.sub(r"[^a-z0-9\s\+\-\*\/\^\=\(\)\.\,\<\>\≤\≥\≡\∞\π]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _build_vocab(texts: List[str]) -> Tuple[Dict[str, int], int]:
        """
        Build a word → integer index vocabulary from the training corpus.

        Index 0 is reserved for <PAD>, index 1 for <UNK>.
        """
        word_set: set = set()
        for text in texts:
            word_set.update(text.split())

        vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        for idx, word in enumerate(sorted(word_set), start=2):
            vocab[word] = idx

        all_lengths = [len(t.split()) for t in texts]
        max_seq_len = int(np.percentile(all_lengths, 95))
        return vocab, max_seq_len

    @staticmethod
    def _texts_to_sequences(
        texts: List[str],
        vocab: Dict[str, int],
        max_seq_len: int,
    ) -> List[List[int]]:
        """
        Convert text strings to fixed-length integer sequences.

        Sequences longer than *max_seq_len* are truncated; shorter ones
        are right-padded with 0 (PAD index).
        """
        sequences: List[List[int]] = []
        unk_idx = vocab["<UNK>"]
        for text in texts:
            tokens = text.split()[:max_seq_len]
            seq = [vocab.get(tok, unk_idx) for tok in tokens]
            # Pad
            seq += [0] * (max_seq_len - len(seq))
            sequences.append(seq)
        return sequences
