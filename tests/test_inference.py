import pytest
import numpy as np
from unittest.mock import MagicMock

from model.inference import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    Detection,
    InferenceResult,
    draw_detections,
    run_inference,
)



@pytest.fixture
def blank_frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    return Detection(
        class_id=5,
        class_name="plastic",
        confidence=0.87,
        bbox_xyxy=(10, 20, 100, 120),
        bbox_xywhn=(0.17, 0.29, 0.28, 0.42),
    )


@pytest.fixture
def sample_result(sample_detection):
    return InferenceResult(
        detections=[sample_detection],
        inference_ms=12.5,
        frame_width=320,
        frame_height=240,
    )


def test_inference_result_num_detections(sample_result):
    assert sample_result.num_detections == 1


def test_inference_result_detected_classes(sample_result):
    assert "plastic" in sample_result.detected_classes


def test_inference_result_no_error():
    result = InferenceResult()
    assert result.error is None
    assert result.num_detections == 0


def test_run_inference_none_model(blank_frame):
    result = run_inference(None, blank_frame)
    assert result.error is not None
    assert "not loaded" in result.error.lower()


def test_run_inference_empty_frame():
    result = run_inference(MagicMock(), np.array([]))
    assert result.error is not None


def test_draw_detections_returns_ndarray(blank_frame, sample_result):
    annotated = draw_detections(blank_frame, sample_result)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == blank_frame.shape


def test_draw_detections_does_not_modify_original(blank_frame, sample_result):
    original_copy = blank_frame.copy()
    draw_detections(blank_frame, sample_result)
    np.testing.assert_array_equal(blank_frame, original_copy)


def test_draw_detections_empty_result(blank_frame):
    empty_result = InferenceResult(frame_width=320, frame_height=240)
    annotated = draw_detections(blank_frame, empty_result)
    np.testing.assert_array_equal(annotated, blank_frame)



def test_inference_constants():
    assert 0 < CONF_THRESHOLD < 1
    assert 0 < IOU_THRESHOLD  < 1
    assert IMAGE_SIZE > 0

