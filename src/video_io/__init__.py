"""
Video I/O utilities for EcoSort-AI.

Provides helpers to open video sources (files or camera indices),
read frames, and write annotated output videos.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── VideoReader ─────────────────────────────────────────────────────────────

class VideoReader:
    """
    Context-managed wrapper around cv2.VideoCapture.

    Usage
    -----
    with VideoReader("path/to/video.mp4") as reader:
        for frame_idx, frame in reader.frames():
            ...  # process BGR frame
    """

    def __init__(self, source: str | int) -> None:
        self.source = source
        self._cap: Optional[cv2.VideoCapture] = None

    # ── Properties (available after __enter__) ─────────────────────────────

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")
        logger.debug(
            "VideoReader opened: %s  [%dx%d @ %.1f fps, %d frames]",
            self.source, self.width, self.height, self.fps, self.total_frames,
        )
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── Frame iterator ─────────────────────────────────────────────────────

    def frames(self, skip: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield (frame_index, BGR_frame) for every `skip`-th frame.

        Parameters
        ----------
        skip : int
            Yield one frame every `skip` frames (1 = every frame).
        """
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager.")

        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if idx % skip == 0:
                yield idx, frame
            idx += 1

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame. Returns (success, frame)."""
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None


# ── VideoWriter ─────────────────────────────────────────────────────────────

class VideoWriter:
    """
    Context-managed wrapper around cv2.VideoWriter.

    Usage
    -----
    with VideoWriter("out.mp4", fps=25, width=1280, height=720) as writer:
        writer.write(annotated_frame)
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        fourcc: str = "mp4v",
    ) -> None:
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = fourcc
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def __enter__(self) -> "VideoWriter":
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(
            self.output_path, fourcc_code, self.fps, (self.width, self.height)
        )
        if not self._writer.isOpened():
            raise IOError(f"Cannot open VideoWriter for: {self.output_path}")
        logger.debug("VideoWriter opened: %s  [%dx%d @ %.1f fps]",
                     self.output_path, self.width, self.height, self.fps)
        return self

    def __exit__(self, *_) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None

    def write(self, frame: np.ndarray) -> None:
        """Write a BGR frame to the output file."""
        if self._writer is None:
            raise RuntimeError("VideoWriter must be used as a context manager.")
        self._writer.write(frame)
        self._frames_written += 1


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_video_info(source: str | int) -> dict:
    """
    Return a dict with basic metadata about a video source.

    Keys: fps, width, height, total_frames, duration_s
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return {}
    info = {
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_s"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
    cap.release()
    return info


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a frame to (width, height) preserving aspect ratio with padding."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to exact target size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_off = (height - new_h) // 2
    x_off = (width  - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

