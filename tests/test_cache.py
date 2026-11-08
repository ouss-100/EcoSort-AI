from __future__ import annotations
import numpy as np
from src.cache import InferenceCache, _frame_hash


def _make_frame(value: int = 0, shape=(64, 64, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


class _FakeResult:
    def __init__(self, label: str):
        self.label = label

class TestFrameHash:
    def test_identical_frames_produce_same_hash(self):
        f1 = _make_frame(100)
        f2 = _make_frame(100)
        assert _frame_hash(f1) == _frame_hash(f2)

    def test_different_frames_produce_different_hashes(self):
        f1 = _make_frame(0)
        f2 = _make_frame(255)
        assert _frame_hash(f1) != _frame_hash(f2)

    def test_hash_is_hex_string(self):
        h = _frame_hash(_make_frame())
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)


class TestInferenceCacheGetPut:
    def test_miss_returns_none(self):
        cache = InferenceCache()
        assert cache.get(_make_frame(0)) is None

    def test_put_then_get_returns_result(self):
        cache = InferenceCache()
        frame = _make_frame(42)
        result = _FakeResult("plastic")
        cache.put(frame, result)
        assert cache.get(frame) is result

    def test_put_duplicate_key_is_ignored(self):
        cache = InferenceCache()
        frame = _make_frame(10)
        r1 = _FakeResult("glass")
        r2 = _FakeResult("metal")
        cache.put(frame, r1)
        cache.put(frame, r2)   # duplicate — should not replace
        assert cache.get(frame) is r1

    def test_size_increases_with_puts(self):
        cache = InferenceCache()
        for i in range(5):
            cache.put(_make_frame(i), _FakeResult(f"cls_{i}"))
        assert cache.size == 5


class TestInferenceCacheEviction:
    def test_evicts_oldest_when_full(self):
        cache = InferenceCache(max_size=3)
        frames = [_make_frame(i) for i in range(4)]
        results = [_FakeResult(f"r{i}") for i in range(4)]

        for i in range(3):
            cache.put(frames[i], results[i])

        # Insert a 4th entry — should evict frames[0]
        cache.put(frames[3], results[3])

        assert cache.size == 3
        assert cache.get(frames[0]) is None          # evicted
        assert cache.get(frames[3]) is results[3]    # newest

    def test_size_never_exceeds_max_size(self):
        cache = InferenceCache(max_size=5)
        for i in range(20):
            cache.put(_make_frame(i), _FakeResult(f"r{i}"))
        assert cache.size <= 5


class TestInferenceCacheHitRate:
    def test_zero_ops_gives_zero_hit_rate(self):
        assert InferenceCache().hit_rate == 0.0

    def test_hit_rate_calculated_correctly(self):
        cache = InferenceCache()
        frame = _make_frame(7)
        cache.put(frame, _FakeResult("paper"))

        cache.get(frame)   # hit
        cache.get(frame)   # hit
        cache.get(_make_frame(99))  # miss

        assert cache.hits == 2
        assert cache.misses == 1
        assert abs(cache.hit_rate - round(2 / 3, 3)) < 1e-6


class TestInferenceCacheClear:
    def test_clear_empties_cache(self):
        cache = InferenceCache()
        cache.put(_make_frame(1), _FakeResult("a"))
        cache.put(_make_frame(2), _FakeResult("b"))
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.hit_rate == 0.0

