from __future__ import annotations
from pathlib import Path
import streamlit as st

from app.pages.about import render_about_page
from app.pages.camera import render_camera_page
from app.pages.video import render_video_page
from app.state.session import get_session_summary, init_session_state, reset_session
from config.settings import CONF_THRESHOLD, DEFAULT_WEIGHTS, IOU_THRESHOLD
from core.detection.model import get_model

st.set_page_config(
    page_title="EcoSort AI | Vision Platform",
    page_icon="recycle",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS_PATH = Path(__file__).parent / "app" / "styles" / "theme.css"
st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

WEIGHTS_PATH = DEFAULT_WEIGHTS


@st.cache_resource(show_spinner="Initializing YOLO inference engine...")
def load_model():
    """Load YOLO model once and cache it for the application lifecycle."""
    if not WEIGHTS_PATH.exists():
        return None
    try:
        return get_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


init_session_state()

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.4rem 0 0.8rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    width: 38px;
                    height: 38px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
                ">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
                        <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
                    </svg>
                </div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f9fafb; letter-spacing: -0.02em;">EcoSort AI</div>
                    <div style="font-size: 0.75rem; color: #9ca3af; font-weight: 500;">Vision Platform</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("##### Navigation")
    page = st.radio(
        label="Navigation Menu",
        options=["Live Camera", "Video Analysis", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("##### Engine Status")
    if WEIGHTS_PATH.exists():
        st.markdown(
            """
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 0.6rem 0.8rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
                    <span style="font-size: 0.82rem; font-weight: 600; color: #f3f4f6;">YOLO26</span>
                </div>
                <span style="
                    font-size: 0.7rem;
                    font-weight: 700;
                    color: #34d399;
                    background: rgba(16, 185, 129, 0.15);
                    border: 1px solid rgba(16, 185, 129, 0.4);
                    padding: 0.15rem 0.5rem;
                    border-radius: 4px;
                ">ONLINE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 0.6rem 0.8rem;
            ">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #f59e0b; display: inline-block;"></span>
                    <span style="font-size: 0.82rem; font-weight: 600; color: #f3f4f6;">YOLO26</span>
                </div>
                <span style="
                    font-size: 0.7rem;
                    font-weight: 700;
                    color: #fbbf24;
                    background: rgba(245, 158, 11, 0.15);
                    border: 1px solid rgba(245, 158, 11, 0.4);
                    padding: 0.15rem 0.5rem;
                    border-radius: 4px;
                ">OFFLINE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Place trained weights at {DEFAULT_WEIGHTS.relative_to(DEFAULT_WEIGHTS.parent.parent.parent)} to activate.")

    st.markdown("---")

    with st.expander("Model Configuration", expanded=False):
        conf = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=CONF_THRESHOLD,
            step=0.05,
            format="%.2f",
            help="Minimum confidence threshold required to register a detection.",
        )
        iou = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.10,
            max_value=0.90,
            value=IOU_THRESHOLD,
            step=0.05,
            format="%.2f",
            help="Non-maximum suppression threshold to filter overlapping bounding boxes.",
        )

    st.markdown("---")

    st.markdown("##### Session Telemetry")
    summary = get_session_summary()
    st.metric("Total Detections", summary["total_detections"])
    st.metric("Frames Analyzed", summary["frames_processed"])

    if st.button("Reset Telemetry", use_container_width=True, key="sidebar_reset"):
        reset_session()
        st.rerun()

    st.markdown("---")
    st.caption("EcoSort-AI v1.0 · YOLO26 Core")

if not WEIGHTS_PATH.exists():
    st.warning(
        f"Model weights not detected at `{DEFAULT_WEIGHTS}`. "
        "Detection pipeline is offline. You can still inspect the platform interface and test modules.",
    )

model = load_model()

if page == "Live Camera":
    render_camera_page(model, conf=conf, iou=iou)
elif page == "Video Analysis":
    render_video_page(model, conf=conf, iou=iou)
else:
    render_about_page()

