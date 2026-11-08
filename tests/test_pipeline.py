from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest
from src.pipline import process_video_file
from src.analytics import SessionReport


def _write_synthetic_video(path: str, num_frames: int = 10, w: int = 64, h: int = 48, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()


def _make_mock_model():
    mock_result = MagicMock()
    mock_result.boxes = None

    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    return mock_model


class TestProcessVideoFile:
    @pytest.fixture()
    def video_paths(self, tmp_path):
        input_path  = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        _write_synthetic_video(input_path, num_frames=10)
        return input_path, output_path

    def test_returns_session_report(self, video_paths):
        input_path, output_path = video_paths
        model = _make_mock_model()

        report = process_video_file(
            input_path=input_path,
            output_path=output_path,
            model=model,
        )

        assert isinstance(report, SessionReport)

    def test_output_file_is_created(self, video_paths):
        input_path, output_path = video_paths
        process_video_file(
            input_path=input_path,
            output_path=output_path,
            model=_make_mock_model(),
        )
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_total_frames_matches_input(self, video_paths):
        input_path, output_path = video_paths
        report = process_video_file(
            input_path=input_path,
            output_path=output_path,
            model=_make_mock_model(),
        )
        assert report.total_frames >= 10

    def test_frame_skip_reduces_inference_calls(self, video_paths):
        input_path, output_path = video_paths
        model = _make_mock_model()

        report = process_video_file(
            input_path=input_path,
            output_path=output_path,
            model=model,
            frame_skip=5,
        )
        assert report.frames_inferred <= 2

    def test_zero_detections_with_mock_model(self, video_paths):
        input_path, output_path = video_paths
        report = process_video_file(
            input_path=input_path,
            output_path=output_path,
            model=_make_mock_model(),
        )
        assert report.total_detections == 0

    def test_progress_callback_is_called(self, video_paths):
        input_path, output_path = video_paths
        calls = []

        def _cb(current, total):
            calls.append((current, total))

        process_video_file(
            input_path=input_path,
            output_path=output_path,
            model=_make_mock_model(),
            progress_callback=_cb,
        )

        assert len(calls) > 0
        last_current, last_total = calls[-1]
        assert last_current == last_total

    def test_raises_on_missing_input(self, tmp_path):
        with pytest.raises(IOError):
            process_video_file(
                input_path=str(tmp_path / "nonexistent.mp4"),
                output_path=str(tmp_path / "out.mp4"),
                model=_make_mock_model(),
            )

    def test_report_is_finished(self, video_paths):
        input_path, output_path = video_paths
        report = process_video_file(
            input_path=input_path,
            output_path=output_path,
            model=_make_mock_model(),
        )
        assert report.finished_at > 0
        assert report.duration_s > 0

