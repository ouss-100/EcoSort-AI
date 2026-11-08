"""
Cache module — lightweight in-memory caching for inference results.

Avoids redundant inference on identical frames (e.g. static camera scenes).
"""
from __future__ import annotations

import hashlib
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


def _frame_hash(frame: np.ndarray) -> str:
    """Compute a fast hash of a frame using strided sampling (much faster than full MD5)."""
    # Sample every 50th pixel in both axes — cheap and collision-resistant enough for caching
    sample = frame[::50, ::50].tobytes()
    # usedforsecurity=False: MD5 is used only as a cache key, not for security
    return hashlib.md5(sample, usedforsecurity=False).hexdigest()


class InferenceCache:
    """
    LRU-style in-memory cache for InferenceResult objects.

    Keyed by a fast perceptual hash of the raw frame so identical frames
    (e.g. a frozen webcam) are not re-inferred.

    Parameters
    ----------
    max_size : maximum number of results to keep in memory
    """

    def __init__(self, max_size: int = 128) -> None:
        self._cache: dict = {}
        self._order: deque = deque()   # PERF-01: O(1) popleft instead of O(n) pop(0)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, frame: np.ndarray):
        """Return cached InferenceResult or None if not found."""
        key = _frame_hash(frame)
        if key in self._cache:
            self.hits += 1
            logger.debug("Cache HIT  (hits=%d, misses=%d)", self.hits, self.misses)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, frame: np.ndarray, result) -> None:
        """Store an InferenceResult for the given frame."""
        key = _frame_hash(frame)
        if key in self._cache:
            return
        if len(self._order) >= self.max_size:
            oldest = self._order.popleft()   # O(1) with deque
            self._cache.pop(oldest, None)
        self._cache[key] = result
        self._order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total > 0 else 0.0

