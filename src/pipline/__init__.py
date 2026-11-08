"""
Pipeline orchestrator for EcoSort-AI.

Ties together VideoReader, DetectionPipeline, VideoWriter,
and analytics into a single high-level API.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Callable, Optional
from src.analytics import SessionReport
from src.detection import DetectionPipeline
from src.video_io import VideoReader, VideoWriter, get_video_info
import cv2


logger = logging.getLogger(__name__)


def process_video_file(
    input_path: str | Path,
    output_path: str | Path,
    model,
    conf: float = 0.35,
    iou:  float = 0.45,
    frame_skip: int = 1,
    annotate:   bool = True,
    target_size: Optional[tuple[int, int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SessionReport:
    """
    Process a video file end-to-end.

    Reads input_path, runs detection on every `frame_skip`-th frame,
    writes annotated frames to output_path, and returns a SessionReport.

    Parameters
    ----------
    input_path        : path to the source video
    output_path       : path for the annotated output video
    model             : loaded YOLO model
    conf              : confidence threshold
    iou               : IoU / NMS threshold
    frame_skip        : run inference every N frames (1 = every frame)
    annotate          : whether to draw bounding boxes on output frames
    target_size       : (width, height) to resize output, or None = original
    progress_callback : optional fn(current_frame, total_frames) for progress reporting

    Returns
    -------
    SessionReport with full per-frame analytics.
    """
    
    info = get_video_info(str(input_path))
    if not info:
        raise IOError(f"Cannot read video: {input_path}")

    fps    = info["fps"] or 25.0
    orig_w = info["width"]
    orig_h = info["height"]
    total  = info["total_frames"]

    out_w = target_size[0] if target_size else orig_w
    out_h = target_size[1] if target_size else orig_h

    report  = SessionReport(source=str(input_path))
    pipeline = DetectionPipeline(model, conf=conf, iou=iou,
                                 annotate=annotate, frame_skip=frame_skip)

    t0 = time.perf_counter()

    with VideoReader(str(input_path)) as reader:
        with VideoWriter(str(output_path), fps=fps, width=out_w, height=out_h) as writer:
            for frame_idx, frame in reader.frames():
                if target_size:
                    frame = cv2.resize(frame, (out_w, out_h))

                result, out_frame = pipeline.process_frame(frame)

                if result and not result.error:
                    report.record_frame(frame_idx, result)
                else:
                    report.total_frames += 1  # count even skipped frames

                writer.write(out_frame)

                if progress_callback:
                    progress_callback(frame_idx + 1, total)

    report.finish()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Processed %d frames in %.1fs (%.1f fps avg) — %d detections",
        report.total_frames, elapsed, report.avg_fps, report.total_detections,
    )
    return report

