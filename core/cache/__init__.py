from __future__ import annotations

import hashlib
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


def _frame_hash(frame: np.ndarray) -> str:
    sample = frame[::50, ::50].tobytes()
    return hashlib.md5(sample, usedforsecurity=False).hexdigest()


class InferenceCache:
    def __init__(self, max_size: int = 128) -> None:
        self._cache: dict = {}
        self._order: deque = deque()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, frame: np.ndarray):
        key = _frame_hash(frame)
        if key in self._cache:
            self.hits += 1
            logger.debug("Cache HIT  (hits=%d, misses=%d)", self.hits, self.misses)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, frame: np.ndarray, result) -> None:
        key = _frame_hash(frame)
        if key in self._cache:
            return
        if len(self._order) >= self.max_size:
            oldest = self._order.popleft()
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

