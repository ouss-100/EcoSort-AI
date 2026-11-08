from src.tracking import CentroidTracker


def test_empty_update_returns_empty():
    tracker = CentroidTracker()
    result = tracker.update([])
    assert len(result) == 0


def test_single_object_gets_id():
    tracker = CentroidTracker()
    objects = tracker.update([(10, 10, 50, 50)])
    assert len(objects) == 1
    obj_id = list(objects.keys())[0]
    assert obj_id == 0


def test_same_object_keeps_id():
    tracker = CentroidTracker()
    objects1 = tracker.update([(10, 10, 50, 50)])
    id1 = list(objects1.keys())[0]
    # Move slightly
    objects2 = tracker.update([(12, 12, 52, 52)])
    id2 = list(objects2.keys())[0]
    assert id1 == id2


def test_disappeared_object_removed():
    tracker = CentroidTracker(max_disappeared=2)
    tracker.update([(10, 10, 50, 50)])
    for _ in range(3):
        tracker.update([])
    assert len(tracker.objects) == 0


def test_new_object_gets_new_id():
    tracker = CentroidTracker()
    objects1 = tracker.update([(10, 10, 50, 50)])
    id1 = list(objects1.keys())[0]
    for _ in range(35):
        objects2 = tracker.update([])
    objects3 = tracker.update([(500, 400, 600, 480)])
    id3 = list(objects3.keys())[0]
    assert id3 != id1


def test_reset_clears_tracker():
    tracker = CentroidTracker()
    tracker.update([(0, 0, 100, 100)])
    tracker.reset()
    assert len(tracker.objects) == 0
    assert tracker.next_object_id == 0

