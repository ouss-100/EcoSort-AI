"""
Rendering utilities for EcoSort-AI.

Extra drawing helpers that complement model/inference.py:
overlays, legends, timestamps, and summary banners.
"""
from __future__ import annotations

import datetime
from typing import Dict, Optional

import cv2
import numpy as np

from model.labels import get_color_bgr
from PIL import Image

def draw_legend(
    frame: np.ndarray,
    class_counts: Dict[str, int],
    position: str = "top-right",
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Draw a semi-transparent legend box showing per-class detection counts.

    Parameters
    ----------
    frame        : BGR numpy array
    class_counts : {class_name: count}
    position     : 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left'
    alpha        : legend box transparency (0=transparent, 1=opaque)
    """
    if not class_counts:
        return frame

    out = frame.copy()
    h, w = frame.shape[:2]

    line_h   = 24
    pad      = 8
    box_w    = 190
    box_h    = pad * 2 + len(class_counts) * line_h + 24

    # Position
    if position == "top-right":
        x1, y1 = w - box_w - 10, 10
    elif position == "top-left":
        x1, y1 = 10, 10
    elif position == "bottom-right":
        x1, y1 = w - box_w - 10, h - box_h - 10
    else:  # bottom-left
        x1, y1 = 10, h - box_h - 10

    x2, y2 = x1 + box_w, y1 + box_h

    # Semi-transparent background
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # Title
    cv2.putText(out, "Detections", (x1 + pad, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    y_cursor = y1 + 24 + pad
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        color = get_color_bgr(cls_name)
        # Color swatch
        cv2.rectangle(out, (x1 + pad, y_cursor), (x1 + pad + 14, y_cursor + 14), color, cv2.FILLED)
        label = f"{cls_name}: {count}"
        cv2.putText(out, label, (x1 + pad + 20, y_cursor + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y_cursor += line_h

    return out


def draw_timestamp(frame: np.ndarray, text: Optional[str] = None) -> np.ndarray:
    """
    Overlay a timestamp in the bottom-left corner.

    Parameters
    ----------
    frame : BGR numpy array
    text  : custom text; defaults to current UTC time string
    """
    out = frame.copy()
    ts = text or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    h  = frame.shape[0]
    cv2.putText(
        out, ts, (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )
    return out


def draw_fps_counter(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS counter in the top-left corner."""
    out = frame.copy()
    cv2.putText(
        out, f"FPS: {fps:.1f}", (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return out


def frame_to_rgb(bgr_frame: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV frame to RGB (for Streamlit / PIL display)."""
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)


def bgr_to_pil(bgr_frame: np.ndarray):
    """Convert a BGR numpy frame to a PIL Image."""
    
    return Image.fromarray(frame_to_rgb(bgr_frame))

