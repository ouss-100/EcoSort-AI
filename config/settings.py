from __future__ import annotations
from pathlib import Path

ROOT_DIR      = Path(__file__).parent.parent
MODELS_DIR    = ROOT_DIR / "models"
DATA_DIR      = ROOT_DIR / "data"
LOGS_DIR      = ROOT_DIR / "logs"
EXPORTS_DIR   = ROOT_DIR / "exports"

DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"

CONF_THRESHOLD  = 0.35   # minimum detection confidence
IOU_THRESHOLD   = 0.45   # NMS IoU threshold
IMAGE_SIZE      = 640    # YOLO input size (px)

DEFAULT_FPS       = 25
MAX_UPLOAD_MB     = 500
SUPPORTED_VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

