"""
Models module — src-level model management utilities.

Provides higher-level wrappers for loading, validating, and
switching between model weight files at runtime.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
from model.model import get_model

logger = logging.getLogger(__name__)


def find_weight_files(search_dirs: Optional[List[str | Path]] = None) -> List[Path]:
    """
    Search for .pt weight files in common locations.

    Parameters
    ----------
    search_dirs : directories to search (defaults to ./models and ./runs)

    Returns
    -------
    Sorted list of .pt file paths found.
    """
    if search_dirs is None:
        root = Path(__file__).parent.parent.parent
        search_dirs = [root / "models", root / "runs"]

    found = []
    for d in search_dirs:
        d = Path(d)
        if d.exists():
            found.extend(sorted(d.rglob("*.pt")))
    return found


def load_model_safe(weights_path: str | Path):
    """
    Load a YOLO model from weights_path with full error reporting.

    Returns the model on success or None on failure.
    """
    
    path = Path(weights_path)
    if not path.exists():
        logger.error("Weights file not found: %s", path)
        return None
    return get_model(str(path))


def validate_weights(weights_path: str | Path) -> bool:
    """Return True if the file exists and ends in .pt."""
    p = Path(weights_path)
    return p.exists() and p.suffix == ".pt"

