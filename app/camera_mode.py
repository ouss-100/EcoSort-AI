import cv2
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

from app.state.session import get_session_summary, init_session_state, reset_session, update_stats
from app.components.icons import SVG_ICONS
from app.components.header import render_header
from app.components.metrics import render_metric_row
from app.components.charts import render_class_chart
from app.components.tables import render_detection_table
from app.components.badges import category_badges_html
from core.detection.inference import draw_detections, run_inference
from core.detection.labels import CLASS_COLORS_HEX, CLASSES


def _init_stream_state() -> None:
    defaults = {
        "stream_active":    False,
        "stream_cam_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _open_capture(cam_index: int) -> bool:
    """
    Open the camera and store it in session_state['_cap_obj'].
    Returns True on success, False if the camera could not be opened.
    """
    _release_capture()
    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    st.session_state["_cap_obj"] = cap
    return True


def _release_capture() -> None:
    """Release and discard any open VideoCapture object."""
    cap = st.session_state.pop("_cap_obj", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def _read_frame():
    """
    Read one frame from the persistent capture object.
    Returns (ret: bool, frame: ndarray | None).
    """
    cap = st.session_state.get("_cap_obj")
    if cap is None or not cap.isOpened():
        return False, None
    for _ in range(2):
        cap.grab()
    ret, frame = cap.retrieve()
    return ret, frame


@st.fragment(run_every=0.067)
def _stream_fragment(model, conf: float, iou: float) -> None:
    """
    Isolated fragment that refreshes every ~67 ms.
    When streaming is inactive it shows a static placeholder — no flicker.
    """
    if not st.session_state.get("stream_active", False):
        st.markdown(
            f"""
            <div style="
                height: 400px;
                border: 2px dashed #1f2937;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #6b7280;
                font-size: 0.9rem;
                flex-direction: column;
                gap: 0.75rem;
                background: #111827;
            ">
                <span style="opacity: 0.55;">{SVG_ICONS['camera']}</span>
                <span>Press <strong style='color:#f9fafb;'>Start Stream</strong> to begin live detection</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    ret, frame_bgr = _read_frame()

    if not ret or frame_bgr is None:
        opened = _open_capture(st.session_state.stream_cam_index)
        if not opened:
            st.error(
                "Cannot open camera. "
                "Check the camera index or grant OS permissions, then restart the stream."
            )
            st.session_state.stream_active = False
            return
        ret, frame_bgr = _read_frame()
        if not ret or frame_bgr is None:
            st.warning("Waiting for first frame…")
            return

    if model is not None:
        result = run_inference(model, frame_bgr, conf=conf, iou=iou)
        if result.error:
            st.error(f"Inference error: {result.error}")
            return

        update_stats(result)
        annotated_bgr = draw_detections(frame_bgr, result)
        frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True)

        render_metric_row(
            {
                "Objects Detected": result.num_detections,
                "Inference":        f"{result.inference_ms} ms",
                "Categories":       len(result.detected_classes),
            }
        )

        if result.detected_classes:
            st.markdown(
                category_badges_html(result.detected_classes),
                unsafe_allow_html=True,
            )

        st.markdown("##### Detection Records")
        render_detection_table(result)

    else:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True)
        st.caption("Model offline — displaying raw feed without inference.")


def render_camera_page(model, conf: float, iou: float) -> None:
    init_session_state()
    _init_stream_state()

    render_header(
        title="Live Camera Detection",
        subtitle="Real-time optical feed analysis with YOLO26 inference engine",
    )

    if model is None:
        st.warning("Model offline — camera will display raw feed without inference.")

    ctrl_left, ctrl_mid, ctrl_right = st.columns([1, 1, 2], gap="small")

    with ctrl_left:
        cam_index = st.number_input(
            "Camera index",
            min_value=0,
            max_value=8,
            value=st.session_state.stream_cam_index,
            step=1,
            help="0 = built-in webcam, 1+ = external cameras",
            disabled=st.session_state.stream_active,
        )
        st.session_state.stream_cam_index = int(cam_index)

    with ctrl_mid:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if not st.session_state.stream_active:
            if st.button(
                "Start Stream", type="primary",
                use_container_width=True, key="btn_start",
            ):
                opened = _open_capture(st.session_state.stream_cam_index)
                if opened:
                    st.session_state.stream_active = True
                else:
                    st.error(
                        f"Could not open camera {st.session_state.stream_cam_index}. "
                        "Try a different index or check permissions."
                    )
                st.rerun()
        else:
            if st.button(
                "Stop Stream", type="secondary",
                use_container_width=True, key="btn_stop",
            ):
                st.session_state.stream_active = False
                _release_capture()
                st.rerun()

    with ctrl_right:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if st.session_state.stream_active:
            st.markdown(
                """<div style="
                    display:flex; align-items:center; gap:0.5rem;
                    background:rgba(16,185,129,0.1);
                    border:1px solid rgba(16,185,129,0.3);
                    color:#34d399; padding:0.4rem 0.9rem;
                    border-radius:9999px; font-size:0.78rem; font-weight:700;
                    width:fit-content; letter-spacing:0.04em;
                ">
                    <span style="width:8px;height:8px;border-radius:50%;
                                 background:#10b981;display:inline-block;
                                 box-shadow:0 0 8px #10b981;"></span>
                    STREAMING LIVE
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div style="
                    display:flex; align-items:center; gap:0.5rem;
                    background:rgba(107,114,128,0.1); border:1px solid #374151;
                    color:#6b7280; padding:0.4rem 0.9rem;
                    border-radius:9999px; font-size:0.78rem; font-weight:700;
                    width:fit-content; letter-spacing:0.04em;
                ">
                    <span style="width:8px;height:8px;border-radius:50%;
                                 background:#6b7280;display:inline-block;"></span>
                    IDLE
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    _stream_fragment(model, conf, iou)

    st.markdown("---")
    st.markdown("#### Session Analytics")
    _render_session_analytics()


def _render_session_analytics() -> None:
    summary      = get_session_summary()
    class_counts = summary.get("class_counts", {})
    times        = st.session_state.get("inference_times", [])

    render_metric_row({
        "Total Detections":  summary["total_detections"],
        "Frames Analyzed":   summary["frames_processed"],
        "Mean Latency":      f"{summary['avg_inference_ms']} ms",
        "Unique Categories": len([c for c, v in class_counts.items() if v > 0]),
    })

    if summary["total_detections"] == 0:
        st.markdown(
            f"""
            <div style="
                border: 2px dashed #1f2937;
                border-radius: 12px;
                padding: 2.5rem 1.5rem;
                text-align: center;
                background: #111827;
                color: #9ca3af;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.75rem;
                margin-top: 1rem;
            ">
                <span style="opacity: 0.65;">{SVG_ICONS['chart']}</span>
                <span style="font-size: 1rem; font-weight: 700; color: #f9fafb;">
                    No session data yet
                </span>
                <span style="font-size: 0.85rem; color: #6b7280; max-width: 440px;">
                    Start the live stream and run detections to populate analytics,
                    latency benchmarks, and category distributions.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown("---")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Category Distribution")
        _render_class_bar_horizontal(class_counts)

        st.markdown("#### Classification Breakdown")
        _render_class_table(class_counts, summary["total_detections"])

    with col_right:
        st.markdown("#### Inference Latency Timeline")
        _render_latency_timeline(times)

        st.markdown("#### Category Frequency")
        _render_class_bar_vertical(class_counts)

    st.markdown("---")
    st.markdown("#### Export Telemetry Data")
    _render_export(summary)

    st.markdown("---")
    if st.button(
        "Purge Session Telemetry", type="secondary",
        use_container_width=True, key="reset_camera_stream",
    ):
        reset_session()
        st.rerun()


def _render_class_table(class_counts: dict, total: int) -> None:
    rows = []
    for cls in CLASSES:
        count = class_counts.get(cls, 0)
        pct   = 100 * count / total if total > 0 else 0.0
        rows.append({
            "Category":  cls.capitalize(),
            "Count":     count,
            "Share (%)": f"{pct:.1f}%",
        })
    df = pd.DataFrame(rows).sort_values("Count", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_class_bar_horizontal(class_counts: dict) -> None:
    data = {
        cls.capitalize(): class_counts.get(cls, 0)
        for cls in CLASSES
        if class_counts.get(cls, 0) > 0
    }
    if not data:
        st.info("No distribution data available.")
        return
    df = pd.DataFrame.from_dict(data, orient="index", columns=["Detections"])
    st.bar_chart(df, color="#10b981", horizontal=True)


def _render_class_bar_vertical(class_counts: dict) -> None:
    data = {cls.capitalize(): class_counts.get(cls, 0) for cls in CLASSES}
    df   = pd.DataFrame.from_dict(data, orient="index", columns=["Detections"])
    df   = df[df["Detections"] > 0]
    if df.empty:
        st.info("No category data available.")
        return
    st.bar_chart(df, color="#10b981")


def _render_latency_timeline(times: list) -> None:
    if not times:
        st.info("No inference timing records logged.")
        return
    df = pd.DataFrame({"Frame": range(1, len(times) + 1), "Latency (ms)": times})
    df = df.set_index("Frame")
    st.line_chart(df, color="#10b981")
    c1, c2, c3 = st.columns(3)
    c1.metric("Min",  f"{min(times):.1f} ms")
    c2.metric("Mean", f"{sum(times)/len(times):.1f} ms")
    c3.metric("Max",  f"{max(times):.1f} ms")


def _render_export(summary: dict) -> None:
    col_json, col_csv = st.columns(2)

    report = {
        "exported_at":      datetime.now(timezone.utc).isoformat(),
        "total_detections": summary["total_detections"],
        "frames_processed": summary["frames_processed"],
        "avg_inference_ms": summary["avg_inference_ms"],
        "class_counts":     summary["class_counts"],
    }
    json_str = json.dumps(report, indent=2)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with col_json:
        st.download_button(
            label="Download JSON Telemetry",
            data=json_str,
            file_name=f"ecosort_telemetry_{ts}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json_cam",
        )

    rows    = [{"class": cls, "count": summary["class_counts"].get(cls, 0)} for cls in CLASSES]
    csv_str = pd.DataFrame(rows).to_csv(index=False)
    with col_csv:
        st.download_button(
            label="Download CSV Dataset",
            data=csv_str,
            file_name=f"ecosort_classes_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_csv_cam",
        )
