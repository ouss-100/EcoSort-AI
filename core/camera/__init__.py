from __future__ import annotations
import logging
import time
from typing import List, Optional
import cv2
import numpy as np
from core.detection.inference import draw_detections, run_inference
from core.rendering import draw_fps_counter, draw_legend

logger = logging.getLogger(__name__)


def list_cameras(max_index: int = 5) -> List[int]:
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def capture_frame(camera_index: int = 0) -> Optional[np.ndarray]:
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

            now = time.perf_counter()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now
            frame = draw_fps_counter(frame, fps)
            frame = draw_legend(frame, class_counts)

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(quit_key) or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time detection loop ended.")

