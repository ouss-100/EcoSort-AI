from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np
from config.settings import CONF_THRESHOLD, IMAGE_SIZE, IOU_THRESHOLD
from model.labels import ID_TO_CLASS, get_color_bgr
import time

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id:    int
    class_name:  str
    confidence:  float
    bbox_xyxy:   Tuple[int, int, int, int]
    bbox_xywhn:  Tuple[float, float, float, float]


@dataclass
class InferenceResult:
    detections:     List[Detection] = field(default_factory=list)
    inference_ms:   float = 0.0
    frame_width:    int = 0
    frame_height:   int = 0
    error:          Optional[str] = None

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def detected_classes(self) -> List[str]:
        return list({d.class_name for d in self.detections})


def run_inference(model, frame: np.ndarray, conf: float = CONF_THRESHOLD, iou:  float = IOU_THRESHOLD) -> InferenceResult:
    if model is None:
        return InferenceResult(error="Model is not loaded.")

    if frame is None or frame.size == 0:
        return InferenceResult(error="Empty or invalid frame.")

    h, w = frame.shape[:2]

    try:
        t0 = time.perf_counter()
        results = model.predict(source=frame, conf=conf, iou=iou, imgsz=IMAGE_SIZE, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = results[0]
        detections: List[Detection] = []

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id   = int(boxes.cls[i].item())
                conf_val = float(boxes.conf[i].item())
                # Absolute pixel coords
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].cpu().numpy()]
                # Normalised centre format
                cx, cy, bw, bh  = [float(v) for v in boxes.xywhn[i].cpu().numpy()]

                class_name = ID_TO_CLASS.get(cls_id, f"class_{cls_id}")
                detections.append(Detection(
                    class_id   = cls_id,
                    class_name = class_name,
                    confidence = round(conf_val, 4),
                    bbox_xyxy  = (x1, y1, x2, y2),
                    bbox_xywhn = (round(cx, 4), round(cy, 4),
                                  round(bw, 4), round(bh, 4)),
                ))

        return InferenceResult(
            detections   = detections,
            inference_ms = round(elapsed_ms, 1),
            frame_width  = w,
            frame_height = h,
        )

    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        return InferenceResult(error=str(exc))


def draw_detections(
    frame: np.ndarray,
    result: InferenceResult,
    line_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    annotated = frame.copy()

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        color = get_color_bgr(det.class_name)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Build label text
        label = f"{det.class_name} {det.confidence:.2f}"

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
        )
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        label_y2 = label_y1 + text_h + baseline + 4

        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + text_w + 4, label_y2),
            color,
            cv2.FILLED,
        )

        # Draw label text (white on coloured background)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_thickness,
            cv2.LINE_AA,
        )

    return annotated
