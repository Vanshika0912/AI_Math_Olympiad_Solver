"""
Data Ingestion Component
========================
Responsibilities
----------------
1. Load the raw Math Olympiad CSV dataset from the configured source path.
2. Validate schema and handle missing values.
3. Persist the cleaned raw data to ``artifacts/data/raw_dataset.csv``.
4. Split into train / test sets and save them to ``artifacts/data/``.

All artefact directories are created automatically if they do not exist.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import DataIngestionError
from src.logger import get_logger
from src.utils.common import ensure_directories, load_config

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataIngestionConfig:
    """Typed container for all data-ingestion path settings."""

    source_path: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    target_column: str
    text_column: str
    test_size: float
    random_state: int
    min_text_length: int


class DataIngestion:
    """
    Orchestrates loading, validating, and splitting the dataset.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        cfg = load_config(config_path)
        self.config = DataIngestionConfig(
            source_path=cfg["data"]["source_path"],
            raw_data_path=cfg["paths"]["raw_data_path"],
            train_data_path=cfg["paths"]["train_data_path"],
            test_data_path=cfg["paths"]["test_data_path"],
            target_column=cfg["data"]["target_column"],
            text_column=cfg["data"]["text_column"],
            test_size=cfg["data"]["test_size"],
            random_state=cfg["data"]["random_state"],
            min_text_length=cfg["data"]["min_text_length"],
        )

    # ── public API ───────────────────────────────────────────────────────────

    def initiate(self) -> Tuple[str, str]:
        """
        Execute the full ingestion pipeline.

        Returns
        -------
        (train_path, test_path) : tuple[str, str]
            Paths to the saved train and test CSV files.
        """
        try:
            logger.info("── Data Ingestion started ──")
            df = self._load_data()
            df = self._validate_and_clean(df)
            train_df, test_df = self._split(df)
            train_path, test_path = self._save(df, train_df, test_df)
            logger.info(
                "Data Ingestion complete. Train: %d rows | Test: %d rows",
                len(train_df),
                len(test_df),
            )
            return train_path, test_path
        except DataIngestionError:
            raise
        except Exception as exc:
            raise DataIngestionError(str(exc), sys) from exc

    # ── private helpers ──────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        """Read the source CSV into a DataFrame."""
        source = Path(self.config.source_path)
        if not source.exists():
            raise DataIngestionError(
                f"Source dataset not found at: {source}", sys
            )
        logger.info("Loading dataset from %s", source)
        df = pd.read_csv(source)
        logger.info("Loaded %d rows × %d columns", *df.shape)
        return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and remove invalid rows."""
        required = {self.config.text_column, self.config.target_column}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise DataIngestionError(
                f"Missing required columns: {missing_cols}", sys
            )

        original_len = len(df)

        # Drop rows with null text or label
        df = df.dropna(subset=[self.config.text_column, self.config.target_column])

        # Drop rows where text is too short
        df = df[
            df[self.config.text_column].str.len() >= self.config.min_text_length
        ]

        # Reset index
        df = df.reset_index(drop=True)

        dropped = original_len - len(df)
        if dropped:
            logger.warning("Dropped %d invalid/empty rows during validation.", dropped)

        logger.info(
            "Validation complete. %d rows retained, %d categories.",
            len(df),
            df[self.config.target_column].nunique(),
        )
        return df

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified train/test split."""
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df[self.config.target_column],
        )
        return train_df, test_df

    def _save(
        self,
        raw_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[str, str]:
        """Persist all three DataFrames to the artifact store."""
        ensure_directories(
            [
                Path(self.config.raw_data_path).parent,
                Path(self.config.train_data_path).parent,
                Path(self.config.test_data_path).parent,
            ]
        )

        raw_df.to_csv(self.config.raw_data_path, index=False)
        train_df.to_csv(self.config.train_data_path, index=False)
        test_df.to_csv(self.config.test_data_path, index=False)

        logger.info("Raw data saved  → %s", self.config.raw_data_path)
        logger.info("Train data saved → %s", self.config.train_data_path)
        logger.info("Test data saved  → %s", self.config.test_data_path)

        return self.config.train_data_path, self.config.test_data_path
