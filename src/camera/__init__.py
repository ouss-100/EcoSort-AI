"""
Camera utilities for EcoSort-AI.

Provides helpers for listing available cameras, capturing single
frames from a webcam, and running a real-time detection loop
using OpenCV (non-Streamlit, for CLI/desktop use).
"""
from __future__ import annotations
import logging
import time
from typing import List, Optional
import cv2
import numpy as np
from model.inference import draw_detections, run_inference
from src.rendring import draw_fps_counter, draw_legend

logger = logging.getLogger(__name__)


def list_cameras(max_index: int = 5) -> List[int]:
    """
    Return indices of available camera devices.

    Checks indices 0..max_index-1 and returns those that open successfully.
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def capture_frame(camera_index: int = 0) -> Optional[np.ndarray]:
    """
    Open a camera, grab one frame, and return it as a BGR numpy array.

    Returns None if the camera cannot be opened or the frame is invalid.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.warning("Camera %d could not be opened.", camera_index)
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def realtime_detection_loop(
    model,
    camera_index: int = 0,
    conf: float = 0.35,
    iou:  float = 0.45,
    window_title: str = "EcoSort-AI — Live Detection",
    quit_key: str = "q",
) -> None:
    """
    Run a real-time detection loop using OpenCV window (CLI / desktop use).

    Press `quit_key` (default 'q') or Escape to exit.

    Parameters
    ----------
    model        : loaded YOLO model (from model.model.get_model())
    camera_index : webcam index (0 = default camera)
    conf         : confidence threshold
    iou          : NMS IoU threshold
    window_title : OpenCV window name
    quit_key     : keyboard key to stop the loop
    """


    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera index {camera_index}")

    logger.info("Starting real-time detection on camera %d. Press '%s' to quit.", camera_index, quit_key)

    prev_time = time.perf_counter()
    class_counts: dict = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera.")
                break

            result = run_inference(model, frame, conf=conf, iou=iou)

            if not result.error:
                for det in result.detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                frame = draw_detections(frame, result)

            # FPS overlay
            now = time.perf_counter()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now
            frame = draw_fps_counter(frame, fps)
            frame = draw_legend(frame, class_counts)

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(quit_key) or key == 27:  # 27 = Escape
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time detection loop ended.")

