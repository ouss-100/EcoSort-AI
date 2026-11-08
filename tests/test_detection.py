import numpy as np
import torch
from unittest.mock import MagicMock
from src.detection import DetectionPipeline, PipelineStats


def _blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mock_model_with_detection(class_name="plastic"):
    box = MagicMock()
    box.cls = [torch.tensor(5)]
    box.conf = [torch.tensor(0.88)]

    xyxy = MagicMock()
    xyxy.__getitem__ = MagicMock(return_value=torch.tensor([10.0, 20.0, 100.0, 120.0]))
    box.xyxy = xyxy

    xywhn = MagicMock()
    xywhn.__getitem__ = MagicMock(return_value=torch.tensor([0.17, 0.29, 0.28, 0.42]))
    box.xywhn = xywhn

    result0 = MagicMock()
    result0.boxes = box

    model = MagicMock()
    model.predict.return_value = [result0]
    return model


def test_pipeline_stats_initial():
    stats = PipelineStats()
    assert stats.total_frames == 0
    assert stats.avg_fps == 0.0
    assert stats.avg_inference_ms == 0.0


def test_pipeline_none_model_skips_inference():
    pipeline = DetectionPipeline(model=None, annotate=False)
    result, frame = pipeline.process_frame(_blank_frame())
    assert result is None
    assert pipeline.stats.total_frames == 1


def test_pipeline_frame_skip():
    pipeline = DetectionPipeline(model=None, frame_skip=3)
    for _ in range(9):
        pipeline.process_frame(_blank_frame())
    assert pipeline.stats.total_frames == 9


def test_pipeline_reset():
    pipeline = DetectionPipeline(model=None)
    pipeline.process_frame(_blank_frame())
    assert pipeline.stats.total_frames == 1
    pipeline.reset()
    assert pipeline.stats.total_frames == 0


def test_pipeline_stats_summary_keys():
    pipeline = DetectionPipeline(model=None)
    summary = pipeline.stats.summary()
    for key in ["total_frames", "frames_inferred", "total_detections",
                "avg_fps", "avg_inference_ms"]:
        assert key in summary

