"""
Custom exception hierarchy for AI Math Olympiad Solver.

All pipeline exceptions inherit from ``MathSolverException`` so callers
can catch the base class while still getting fine-grained error types.
"""

import sys
import traceback
from typing import Optional


def _format_error_message(error: Exception, error_detail: sys) -> str:
    """
    Build a rich error message including file name and line number.

    Parameters
    ----------
    error : Exception
        The original exception.
    error_detail : sys
        ``sys`` module passed by the caller to access exc_info.

    Returns
    -------
    str
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return (
            f"Error in [{file_name}] "
            f"at line [{line_number}]: {str(error)}"
        )
    return str(error)


class MathSolverException(Exception):
    """Base exception for the AI Math Olympiad Solver application."""

    def __init__(self, message: str, error_detail: Optional[sys] = None) -> None:
        super().__init__(message)
        if error_detail is not None:
            self.message = _format_error_message(self, error_detail)
        else:
            self.message = message

    def __str__(self) -> str:
        return self.message


class DataIngestionError(MathSolverException):
    """Raised when data loading or validation fails."""


class DataPreprocessingError(MathSolverException):
    """Raised when text cleaning or feature engineering fails."""


class ModelTrainingError(MathSolverException):
    """Raised when model fitting fails."""


class ModelEvaluationError(MathSolverException):
    """Raised when computing evaluation metrics fails."""


class ModelSavingError(MathSolverException):
    """Raised when serialising/deserialising a model fails."""


class InferenceError(MathSolverException):
    """Raised when the inference pipeline cannot produce a prediction."""


class ConfigurationError(MathSolverException):
    """Raised when a required configuration key is missing or invalid."""
