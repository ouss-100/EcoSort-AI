from src.analytics import SessionReport
from model.inference import InferenceResult, Detection


def _make_result(class_name: str, confidence: float = 0.9, inference_ms: float = 15.0):
    det = Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(0, 0, 50, 50),
        bbox_xywhn=(0.1, 0.1, 0.2, 0.2),
    )
    return InferenceResult(
        detections=[det],
        inference_ms=inference_ms,
        frame_width=640,
        frame_height=480,
    )


def test_session_report_initial_state():
    report = SessionReport(source="test.mp4")
    assert report.total_detections == 0
    assert report.total_frames == 0
    assert report.avg_fps == 0.0


def test_session_report_record_frame():
    report = SessionReport(source="test.mp4")
    result = _make_result("plastic")
    report.record_frame(0, result)

    assert report.total_detections == 1
    assert report.frames_inferred == 1
    assert report.class_counts.get("plastic") == 1


def test_session_report_multiple_classes():
    report = SessionReport(source="test.mp4")
    for cls in ["plastic", "glass", "plastic", "metal"]:
        report.record_frame(0, _make_result(cls))

    assert report.total_detections == 4
    assert report.class_counts["plastic"] == 2
    assert report.class_counts["glass"] == 1
    assert report.class_counts["metal"] == 1


def test_session_report_avg_inference():
    report = SessionReport(source="test.mp4")
    for ms in [10.0, 20.0, 30.0]:
        report.record_frame(0, _make_result("cardboard", inference_ms=ms))
    assert report.avg_inference_ms == 20.0


def test_session_report_to_dict_keys():
    report = SessionReport(source="test.mp4")
    d = report.to_dict()
    for key in ["total_frames", "total_detections", "avg_fps",
                "avg_inference_ms", "class_counts", "duration_s"]:
        assert key in d, f"Missing key: {key}"


def test_session_report_finish():
    report = SessionReport(source="test.mp4")
    assert report.finished_at == 0.0
    report.finish()
    assert report.finished_at > 0.0
    assert report.duration_s >= 0.0

