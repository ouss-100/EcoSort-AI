"""
Analytics module for EcoSort-AI.

Provides per-session and per-video aggregation of detection statistics,
suitable for export to CSV or display in the Streamlit dashboard.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from model.labels import CLASSES


@dataclass
class FrameRecord:
    """Statistics for a single processed frame."""
    frame_idx:       int
    timestamp_s:     float
    num_detections:  int
    inference_ms:    float
    class_counts:    Dict[str, int]


@dataclass
class SessionReport:
    """Full analytics report for a processing session."""
    source:             str = ""
    started_at:         float = field(default_factory=time.time)
    finished_at:        float = 0.0
    total_frames:       int = 0
    frames_inferred:    int = 0
    total_detections:   int = 0
    class_counts:       Dict[str, int] = field(default_factory=dict)
    frame_records:      List[FrameRecord] = field(default_factory=list)
    inference_times_ms: List[float] = field(default_factory=list)

    # ── Computed properties ────────────────────────────────────────────────

    @property
    def duration_s(self) -> float:
        if self.finished_at > 0:
            return round(self.finished_at - self.started_at, 2)
        return round(time.time() - self.started_at, 2)

    @property
    def avg_fps(self) -> float:
        d = self.duration_s
        return round(self.total_frames / d, 2) if d > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        if not self.inference_times_ms:
            return 0.0
        return round(sum(self.inference_times_ms) / len(self.inference_times_ms), 1)

    @property
    def detections_per_frame(self) -> float:
        if self.frames_inferred == 0:
            return 0.0
        return round(self.total_detections / self.frames_inferred, 2)

    # ── Mutators ───────────────────────────────────────────────────────────

    def record_frame(self, frame_idx: int, result) -> None:
        """Ingest an InferenceResult for one frame."""
        ts = time.time() - self.started_at
        counts = defaultdict(int)
        for det in result.detections:
            counts[det.class_name] += 1
            self.class_counts[det.class_name] = (
                self.class_counts.get(det.class_name, 0) + 1
            )
        self.total_frames += 1
        self.frames_inferred += 1
        self.total_detections += result.num_detections
        if result.inference_ms > 0:
            self.inference_times_ms.append(result.inference_ms)
        self.frame_records.append(FrameRecord(
            frame_idx=frame_idx,
            timestamp_s=round(ts, 3),
            num_detections=result.num_detections,
            inference_ms=result.inference_ms,
            class_counts=dict(counts),
        ))

    def finish(self) -> None:
        """Mark the session as finished."""
        self.finished_at = time.time()

    # ── Export ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "source":             self.source,
            "duration_s":         self.duration_s,
            "total_frames":       self.total_frames,
            "frames_inferred":    self.frames_inferred,
            "total_detections":   self.total_detections,
            "avg_fps":            self.avg_fps,
            "avg_inference_ms":   self.avg_inference_ms,
            "detections_per_frame": self.detections_per_frame,
            "class_counts":       self.class_counts,
        }

    def to_json(self, path: str | Path) -> None:
        """Save summary to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def frames_to_csv(self, path: str | Path) -> None:
        """Save per-frame records to a CSV file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame_idx", "timestamp_s", "num_detections",
                            "inference_ms"] + CLASSES,
            )
            writer.writeheader()
            for rec in self.frame_records:
                row = {
                    "frame_idx":       rec.frame_idx,
                    "timestamp_s":     rec.timestamp_s,
                    "num_detections":  rec.num_detections,
                    "inference_ms":    rec.inference_ms,
                }
                for cls in CLASSES:
                    row[cls] = rec.class_counts.get(cls, 0)
                writer.writerow(row)

