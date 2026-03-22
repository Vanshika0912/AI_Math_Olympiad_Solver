"""
Shared utility helpers used across the entire pipeline.

Responsibilities
----------------
- YAML configuration loading
- Artifact directory creation
- Serialisation (save / load) of any Python object via joblib
- Metric formatting helpers
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import yaml

from src.exception import ConfigurationError, ModelSavingError
from src.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Load and return the YAML configuration as a nested dict.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML file (relative to project root).

    Returns
    -------
    dict
    """
    path = Path(config_path)
    try:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        logger.info("Configuration loaded from %s", path)
        return cfg
    except Exception as exc:
        raise ConfigurationError(str(exc), sys) from exc


# ─────────────────────────────────────────────────────────────────────────────
# File system helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_directories(dirs: List[Union[str, Path]]) -> None:
    """
    Create a list of directories (including parents) if they do not exist.

    Parameters
    ----------
    dirs : list[str | Path]
        Directory paths to create.
    """
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ensured: %s", directory)


# ─────────────────────────────────────────────────────────────────────────────
# Model / object serialisation
# ─────────────────────────────────────────────────────────────────────────────

def save_object(file_path: Union[str, Path], obj: Any) -> None:
    """
    Serialise *obj* to *file_path* using joblib.

    Parameters
    ----------
    file_path : str | Path
        Destination file (parent directories are created automatically).
    obj : Any
        Python object to serialise.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)
        logger.info("Object saved to %s", path)
    except Exception as exc:
        raise ModelSavingError(str(exc), sys) from exc


def load_object(file_path: Union[str, Path]) -> Any:
    """
    Deserialise and return the object stored at *file_path*.

    Parameters
    ----------
    file_path : str | Path
        Path to a joblib-serialised file.

    Returns
    -------
    Any
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No serialised object at: {path}")
        obj = joblib.load(path)
        logger.info("Object loaded from %s", path)
        return obj
    except Exception as exc:
        raise ModelSavingError(str(exc), sys) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics(metrics: Dict[str, float], model_name: str) -> str:
    """
    Return a pretty-printed metrics summary string.

    Parameters
    ----------
    metrics : dict[str, float]
        Evaluation metrics mapping.
    model_name : str
        Human-readable model identifier.

    Returns
    -------
    str
    """
    lines = [f"\n{'='*55}", f"  Model : {model_name}", f"{'='*55}"]
    for key, value in metrics.items():
        lines.append(f"  {key:<20}: {value:.4f}")
    lines.append(f"{'='*55}\n")
    return "\n".join(lines)
