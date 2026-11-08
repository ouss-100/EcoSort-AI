"""
Detection pipeline module for EcoSort-AI.

Wraps the model inference layer with a higher-level pipeline
that processes batches of frames and returns structured results.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from model.inference import InferenceResult, draw_detections, run_inference

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Aggregate stats for a processed video/session."""
    total_frames:      int   = 0
    frames_inferred:   int   = 0
    total_detections:  int   = 0
    total_time_s:      float = 0.0
    inference_times:   List[float] = field(default_factory=list)
    class_counts:      dict  = field(default_factory=dict)

    @property
    def avg_fps(self) -> float:
        return round(self.total_frames / self.total_time_s, 2) if self.total_time_s > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return round(sum(self.inference_times) / len(self.inference_times), 1)

    def update(self, result: InferenceResult) -> None:
        """Merge an InferenceResult into the running stats."""
        self.frames_inferred += 1
        self.total_detections += result.num_detections
        if result.inference_ms > 0:
            self.inference_times.append(result.inference_ms)
        for det in result.detections:
            self.class_counts[det.class_name] = (
                self.class_counts.get(det.class_name, 0) + 1
            )

    def summary(self) -> dict:
        return {
            "total_frames":     self.total_frames,
            "frames_inferred":  self.frames_inferred,
            "total_detections": self.total_detections,
            "avg_fps":          self.avg_fps,
            "avg_inference_ms": self.avg_inference_ms,
            "total_time_s":     round(self.total_time_s, 2),
            "class_counts":     self.class_counts,
        }


class DetectionPipeline:
    """
    High-level detection pipeline.

    Handles frame-by-frame inference with optional annotation drawing.

    Usage
    -----
    pipeline = DetectionPipeline(model, conf=0.35, iou=0.45)
    result, annotated_frame = pipeline.process_frame(frame)
    print(pipeline.stats.summary())
    """

    def __init__(
        self,
        model,
        conf: float = 0.35,
        iou:  float = 0.45,
        annotate: bool = True,
        frame_skip: int = 1,
    ) -> None:
        self.model      = model
        self.conf       = conf
        self.iou        = iou
        self.annotate   = annotate
        self.frame_skip = frame_skip
        self.stats      = PipelineStats()
        self._frame_idx = 0

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[Optional[InferenceResult], np.ndarray]:
        """
        Run inference on a single frame (if it's not skipped).

        Returns
        -------
        (InferenceResult | None, annotated_frame)
            InferenceResult is None for skipped frames.
        """
        self._frame_idx += 1
        self.stats.total_frames += 1

        should_infer = (self._frame_idx % self.frame_skip == 0) and (self.model is not None)

        if not should_infer:
            return None, frame

        result = run_inference(self.model, frame, conf=self.conf, iou=self.iou)

        if result.error:
            logger.warning("Inference error on frame %d: %s", self._frame_idx, result.error)
            return result, frame

        self.stats.update(result)

        out_frame = draw_detections(frame, result) if self.annotate else frame
        return result, out_frame

    def reset(self) -> None:
        """Reset frame counter and stats (keep model/config)."""
        self.stats = PipelineStats()
        self._frame_idx = 0

