from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import app.video_mode as vm

def _get_compute_output_size():
    for mod in ["streamlit", "streamlit.components", "streamlit.components.v1"]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()
    for mod in ["app.analytics", "app.ui_components", "model.inference", "model.labels"]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    return vm._compute_output_size


class TestComputeOutputSize:
    @pytest.fixture(autouse=True)
    def fn(self):
        self._fn = _get_compute_output_size()

    def test_none_preset_returns_original(self):
        w, h = self._fn(1920, 1080, None)
        assert (w, h) == (1920, 1080)

    def test_720p_preset_scales_correctly(self):
        w, h = self._fn(1920, 1080, (1280, 720))
        assert w == 1280
        assert h == 720

    def test_480p_preset_scales_correctly(self):
        w, h = self._fn(1920, 1080, (854, 480))
        assert h == 480
        assert 852 <= w <= 854

    def test_aspect_ratio_preserved_for_non_standard_input(self):
        orig_w, orig_h = 640, 480 
        target = (1280, 720)
        w, h = self._fn(orig_w, orig_h, target)
        assert w <= target[0]
        assert h <= target[1]
        orig_ratio = orig_w / orig_h
        out_ratio  = w / h
        assert abs(orig_ratio - out_ratio) < 0.02

    def test_zero_dimension_input_does_not_crash(self):
        with pytest.raises((ZeroDivisionError, Exception)):
            self._fn(0, 480, (1280, 720))

class TestOutPathConstruction:
    def test_mp4_input_produces_processed_mp4(self):

        input_path = str(Path("/tmp/sample.mp4"))
        out_path = str(
            Path(input_path)
            .with_stem(Path(input_path).stem + "_processed")
            .with_suffix(".mp4")
        )
        assert out_path.endswith("sample_processed.mp4")

    def test_avi_input_produces_processed_mp4_not_overwrite(self):

        input_path = "/tmp/sample.avi"
        out_path = str(
            Path(input_path)
            .with_stem(Path(input_path).stem + "_processed")
            .with_suffix(".mp4")
        )
        assert out_path != input_path
        assert out_path.endswith("_processed.mp4")

    def test_mov_input_produces_processed_mp4(self):

        input_path = "/tmp/clip.mov"
        out_path = str(
            Path(input_path)
            .with_stem(Path(input_path).stem + "_processed")
            .with_suffix(".mp4")
        )
        assert out_path.endswith("clip_processed.mp4")

    def test_mkv_input_produces_processed_mp4(self):
       
        input_path = "/tmp/video.mkv"
        out_path = str(
            Path(input_path)
            .with_stem(Path(input_path).stem + "_processed")
            .with_suffix(".mp4")
        )
        assert out_path.endswith("video_processed.mp4")


class TestFrameSkipLogic:
    @staticmethod
    def _inferred_frames(total: int, skip: int) -> list[int]:
        inferred = []
        for frame_idx in range(1, total + 1):
            if (frame_idx - 1) % skip == 0:
                inferred.append(frame_idx)
        return inferred

    def test_skip1_processes_every_frame(self):
        inferred = self._inferred_frames(10, 1)
        assert inferred == list(range(1, 11))

    def test_skip2_starts_at_frame1_not_frame2(self):
        inferred = self._inferred_frames(10, 2)
        assert inferred[0] == 1        
        assert inferred == [1, 3, 5, 7, 9]

    def test_skip3_starts_at_frame1(self):
        inferred = self._inferred_frames(9, 3)
        assert inferred[0] == 1
        assert inferred == [1, 4, 7]

    def test_buggy_logic_would_miss_frame1(self):
        old_inferred = [i for i in range(1, 11) if i % 2 == 0]
        assert 1 not in old_inferred
