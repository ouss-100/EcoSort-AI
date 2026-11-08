from __future__ import annotations
import numpy as np
import pytest
from src.rendring import draw_fps_counter, draw_legend, draw_timestamp


def _blank(h=240, w=320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDrawLegend:
    def test_returns_frame_unchanged_when_empty_counts(self):
        frame = _blank()
        result = draw_legend(frame, {})
        assert result is frame

    def test_returns_ndarray_with_counts(self):
        frame = _blank()
        result = draw_legend(frame, {"plastic": 3, "glass": 1})
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_does_not_modify_original_frame(self):
        frame = _blank()
        original = frame.copy()
        draw_legend(frame, {"paper": 2})
        np.testing.assert_array_equal(frame, original)

    @pytest.mark.parametrize("position", ["top-right", "top-left", "bottom-right", "bottom-left"])
    def test_all_positions_accepted(self, position):
        frame = _blank()
        result = draw_legend(frame, {"metal": 5}, position=position)
        assert result.shape == frame.shape

    def test_multiple_classes(self):
        counts = {
            "biodegradable": 10,
            "cardboard": 3,
            "glass": 0,
            "metal": 7,
            "paper": 2,
            "plastic": 15,
        }
        result = draw_legend(_blank(), counts)
        assert result.shape == (240, 320, 3)

class TestDrawTimestamp:
    def test_returns_ndarray(self):
        result = draw_timestamp(_blank())
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        frame = _blank(480, 640)
        result = draw_timestamp(frame)
        assert result.shape == frame.shape

    def test_custom_text_accepted(self):
        frame = _blank()
        result = draw_timestamp(frame, text="TEST 00:00:00")
        assert result.shape == frame.shape

    def test_does_not_modify_original(self):
        frame = _blank()
        original = frame.copy()
        draw_timestamp(frame, text="hello")
        np.testing.assert_array_equal(frame, original)


class TestDrawFpsCounter:
    def test_returns_ndarray(self):
        result = draw_fps_counter(_blank(), fps=30.0)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        frame = _blank(480, 640)
        result = draw_fps_counter(frame, fps=25.5)
        assert result.shape == frame.shape

    def test_zero_fps_accepted(self):
        frame = _blank()
        result = draw_fps_counter(frame, fps=0.0)
        assert result.shape == frame.shape

    def test_does_not_modify_original(self):
        frame = _blank()
        original = frame.copy()
        draw_fps_counter(frame, fps=15.0)
        np.testing.assert_array_equal(frame, original)

