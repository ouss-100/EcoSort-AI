"""
Simple centroid tracker for multi-object tracking across frames.

Assigns stable IDs to detected objects by matching centroids between
consecutive frames using Euclidean distance.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)

class CentroidTracker:
    """
    Lightweight centroid-based multi-object tracker.

    Objects are matched frame-to-frame by nearest centroid.
    Objects that disappear for `max_disappeared` frames are deregistered.

    Usage
    -----
    tracker = CentroidTracker(max_disappeared=30)
    for frame in frames:
        result = run_inference(model, frame)
        rects = [(d.bbox_xyxy[0], d.bbox_xyxy[1],
                  d.bbox_xyxy[2], d.bbox_xyxy[3]) for d in result.detections]
        objects = tracker.update(rects)
        for obj_id, centroid in objects.items():
            ...
    """

    def __init__(self, max_disappeared: int = 30) -> None:
        self.next_object_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.disappeared: OrderedDict[int, int]   = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(
        self, rects: List[Tuple[int, int, int, int]]
    ) -> OrderedDict[int, np.ndarray]:
        """
        Update tracker with new bounding boxes.

        Parameters
        ----------
        rects : list of (x1, y1, x2, y2) bounding boxes

        Returns
        -------
        OrderedDict mapping object_id -> centroid (x, y)
        """
        if not rects:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype=int)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = ((x1 + x2) // 2, (y1 + y2) // 2)

        if not self.objects:
            for c in input_centroids:
                self.register(c)
            return self.objects

        object_ids        = list(self.objects.keys())
        object_centroids  = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), input_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects

    def reset(self) -> None:
        """Clear all tracked objects."""
        self.objects.clear()
        self.disappeared.clear()
        self.next_object_id = 0

