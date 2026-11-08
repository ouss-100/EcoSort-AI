import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from app.analytics import init_session_state, reset_session, update_stats
from app.ui_components import (
    SVG_ICONS,
    render_class_chart,
    render_detection_table,
    render_header,
    render_metric_row,
)
from model.inference import draw_detections, run_inference
from model.labels import CLASS_COLORS_HEX, CLASSES


def render_camera_page(model, conf: float, iou: float) -> None:
    init_session_state()

    render_header(
        title="Live Camera Detection",
        subtitle="Real-time optical feed analysis with YOLO26 inference engine",
    )

    st.markdown(
        """
        <div style="
            background: #111827;
            border-left: 4px solid #10b981;
            padding: 0.85rem 1.15rem;
            border-radius: 6px;
            color: #d1d5db;
            font-size: 0.88rem;
            margin-bottom: 1.5rem;
            border-top: 1px solid #1f2937;
            border-right: 1px solid #1f2937;
            border-bottom: 1px solid #1f2937;
        ">
            <strong style="color: #f9fafb;">Workflow:</strong> Position target waste items in the viewfinder, capture a snapshot, then select 
            <strong style="color: #34d399;">Run Detection</strong> to execute neural inference and extract material telemetry.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two-column layout
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Input Feed")
        camera_image = st.camera_input(
            label="Camera viewfinder",
            key="camera_capture",
            help="Grant browser permission to access camera device.",
            label_visibility="collapsed",
        )

        run_btn = st.button(
            "Run Detection",
            type="primary",
            disabled=(camera_image is None or model is None),
            use_container_width=True,
        )

        if model is None:
            st.warning("Model offline — inference pipeline is disabled until weights are loaded.")

    with col_right:
        st.markdown("#### Detection Telemetry")

        if camera_image is not None and run_btn:
            _process_camera_frame(model, camera_image, conf, iou)
        elif camera_image is None:
            st.markdown(
                f"""
                <div style="
                    height: 290px;
                    border: 2px dashed #1f2937;
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #9ca3af;
                    font-size: 0.92rem;
                    flex-direction: column;
                    gap: 0.75rem;
                    background: #111827;
                ">
                    <span style="opacity: 0.7;">{SVG_ICONS['camera']}</span>
                    <span>Capture a frame in the viewfinder to initiate analysis</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Frame captured. Click 'Run Detection' to analyze.")

    # Session stats at the bottom
    st.markdown("---")
    st.markdown("#### Session Cumulative Performance")
    _render_session_stats()



def _process_camera_frame(model, camera_image, conf: float, iou: float) -> None:
    pil_img = Image.open(camera_image)
    frame_rgb = np.array(pil_img)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Executing neural inference..."):
        result = run_inference(model, frame_bgr, conf=conf, iou=iou)

    if result.error:
        st.error(f"Inference pipeline error: {result.error}")
        return

    update_stats(result)

    annotated_bgr = draw_detections(frame_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Annotated Inference Frame", use_container_width=True)

    render_metric_row(
        {
            "Objects Identified": result.num_detections,
            "Latency": f"{result.inference_ms} ms",
            "Categories Found": len(result.detected_classes),
        }
    )

    if result.detected_classes:
        badges = []
        for c in sorted(result.detected_classes):
            color = CLASS_COLORS_HEX.get(c, "#9ca3af")
            badges.append(
                f"""<span style="
                    display: inline-flex;
                    align-items: center;
                    gap: 0.4rem;
                    background: #111827;
                    border: 1px solid #1f2937;
                    padding: 0.3rem 0.65rem;
                    border-radius: 6px;
                    font-size: 0.82rem;
                    font-weight: 600;
                    color: #f3f4f6;
                    margin-right: 0.4rem;
                    margin-bottom: 0.4rem;
                ">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: {color};"></span>
                    {c.capitalize()}
                </span>"""
            )
        st.markdown(
            f"""<div style="margin: 0.75rem 0 1rem 0;">{''.join(badges)}</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("##### Detection Records")
    render_detection_table(result)

    if result.detections:
        st.markdown("##### Frequency by Category")
        render_class_chart(result.detections)


def _render_session_stats() -> None:
    total_det = st.session_state.get("total_detections", 0)
    frames = st.session_state.get("frames_processed", 0)
    class_counts = st.session_state.get("class_counts", {})
    times = st.session_state.get("inference_times", [])
    avg_ms = round(sum(times) / len(times), 1) if times else 0.0

    render_metric_row(
        {
            "Total Objects": total_det,
            "Frames Analyzed": frames,
            "Average Latency": f"{avg_ms} ms",
        }
    )

    if class_counts:
        st.markdown("##### Cumulative Distribution")
        chart_data = {
            cls.capitalize(): class_counts.get(cls, 0)
            for cls in CLASSES
        }
        df = pd.DataFrame.from_dict(chart_data, orient="index", columns=["Detections"])
        df = df[df["Detections"] > 0]
        if not df.empty:
            st.bar_chart(df, horizontal=True, color="#10b981")

    if st.button("Reset Session Stats", key="reset_camera"):
        reset_session()
        st.rerun()
