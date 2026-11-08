from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_model_instance = None
_model_path_loaded: Optional[str] = None

DEFAULT_WEIGHTS = Path(__file__).parent.parent / "models" / "best.pt"


def get_model(weights_path: Optional[str] = None):
    global _model_instance, _model_path_loaded

    target_path = str(weights_path) if weights_path else str(DEFAULT_WEIGHTS)

    if _model_instance is not None and _model_path_loaded == target_path:
        return _model_instance

    try:
        logger.info(f"Loading YOLO26 model from: {target_path}")
        _model_instance = YOLO(target_path)
        _model_path_loaded = target_path
        logger.info("YOLO26 model loaded successfully.")
        return _model_instance

    except FileNotFoundError:
        logger.error(f"Model weights not found: {target_path}")
        return None
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        return None


def is_model_available(weights_path: Optional[str] = None) -> bool:
    target_path = weights_path or str(DEFAULT_WEIGHTS)
    return Path(target_path).is_file()


def reset_model() -> None:
    global _model_instance, _model_path_loaded
    _model_instance = None
    _model_path_loaded = None
