import numpy as np
from src.video_io import get_video_info, resize_frame


def test_get_video_info_invalid_source():
    info = get_video_info("nonexistent_file.mp4")
    assert info == {}


def test_resize_frame_output_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    resized = resize_frame(frame, 320, 240)
    assert resized.shape == (240, 320, 3)


def test_resize_frame_maintains_aspect_ratio():
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    resized = resize_frame(frame, 320, 320)
    assert resized.shape[:2] == (320, 320)


def test_resize_frame_preserves_dtype():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = resize_frame(frame, 50, 50)
    assert resized.dtype == np.uint8

