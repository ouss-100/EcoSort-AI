"""
EcoSort-AI — Main Streamlit application entry point.
Enterprise computer vision platform for waste classification.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

# ── Page config (must be the very first Streamlit call) ────────────────────
st.set_page_config(
    page_title="EcoSort AI | Vision Platform",
    page_icon="recycle",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Enterprise Dark Theme & Custom CSS ─────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Global Typography */
    html, body, p, h1, h2, h3, h4, h5, h6, label, input, textarea {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Force Full Dark Canvas Across All Containers (Eliminates Browser Light Cache) */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    .main,
    .main .block-container,
    [data-testid="stVerticalBlock"],
    [data-testid="stMainBlockContainer"] {
        background-color: #0b0f19 !important;
        color: #f3f4f6 !important;
    }

    [data-testid="stHeader"] {
        border-bottom: 1px solid #1f2937 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarNav"],
    section[data-testid="stSidebar"] {
        background-color: #080c14 !important;
        border-right: 1px solid #1f2937 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #1f2937 !important;
        margin: 1.2rem 0;
    }

    /* Preserve and enforce Streamlit Material Symbols Icons */
    [data-testid="stIconMaterial"],
    [data-testid*="stIcon"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] button,
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    [data-testid="stExpander"] summary span[data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-style: normal !important;
        font-weight: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
    }

    /* Layout Spacing */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1240px;
    }

    /* Navigation Radio Items */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.35rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.6rem 0.85rem;
        border-radius: 8px;
        font-size: 0.92rem;
        font-weight: 500;
        color: #9ca3af;
        border: 1px solid transparent;
        transition: all 0.15s ease;
        background: transparent;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #1f2937;
        color: #f3f4f6;
    }

    /* Expander Layout */
    [data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    [data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 0.65rem 0.85rem !important;
        cursor: pointer !important;
    }

    [data-testid="stExpander"] summary span {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #f3f4f6 !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #9ca3af !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #f9fafb !important;
        letter-spacing: -0.03em !important;
    }

    /* Primary Action Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: 1px solid #10b981;
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.92rem;
        border-radius: 8px;
        padding: 0.6rem 1.25rem;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.45);
        transform: translateY(-1px);
    }

    /* Secondary Action Buttons */
    .stButton > button[kind="secondary"] {
        background: #111827;
        border: 1px solid #374151;
        color: #e5e7eb !important;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.15s ease;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #1f2937;
        border-color: #4b5563;
        color: #ffffff !important;
    }

    /* Download Buttons */
    .stDownloadButton > button {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399 !important;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 8px;
        transition: all 0.15s ease;
    }

    .stDownloadButton > button:hover {
        background: rgba(16, 185, 129, 0.22);
        border-color: #10b981;
    }

    /* File Uploader Dropzone Dark Styling */
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploader"] section,
    section[data-testid="stFileUploadDropzone"] {
        background-color: #111827 !important;
        border: 1px dashed #374151 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploadDropzone"]:hover,
    [data-testid="stFileUploader"] section:hover {
        border-color: #10b981 !important;
        background-color: #131d2e !important;
    }

    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] div {
        color: #9ca3af !important;
    }

    [data-testid="stFileUploadDropzone"] button {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
        border: 1px solid #374151 !important;
    }

    /* Selectboxes and Dropdowns */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        color: #f3f4f6 !important;
    }

    div[data-baseweb="input"] input {
        color: #f3f4f6 !important;
    }

    div[data-baseweb="popover"],
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
    }

    li[role="option"] {
        background-color: #111827 !important;
        color: #f3f4f6 !important;
    }

    li[role="option"]:hover,
    li[aria-selected="true"] {
        background-color: #1f2937 !important;
        color: #10b981 !important;
    }

    /* Camera Input Widget */
    [data-testid="stCameraInput"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }

    /* Dataframes and Tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Tabs */
    [data-testid="stTabs"] {
        background: transparent !important;
    }

    [data-testid="stTabs"] button {
        background: transparent !important;
        color: #9ca3af !important;
        font-weight: 500 !important;
        border: none !important;
        padding: 0.6rem 1.25rem !important;
    }

    [data-testid="stTabs"] button:hover {
        color: #f3f4f6 !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #10b981 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #10b981 !important;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background-color: #1f2937;
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Imports (after page config) ────────────────────────────────────────────
from app.about_page import render_about_page
from app.analytics import get_session_summary, init_session_state, reset_session
from app.analytics_page import render_analytics_page
from app.camera_mode import render_camera_page
from app.video_mode import render_video_page
from config.settings import CONF_THRESHOLD, IOU_THRESHOLD

# ── Constants ──────────────────────────────────────────────────────────────
WEIGHTS_PATH = Path("models/best.pt")


# ── Model loader (cached across sessions) ─────────────────────────────────
@st.cache_resource(show_spinner="Initializing YOLO inference engine...")
def load_model():
    """Load YOLO model once and cache it for the application lifecycle."""
    if not WEIGHTS_PATH.exists():
        return None
    try:
        from model.model import get_model
        return get_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


# ── Session state init ─────────────────────────────────────────────────────
init_session_state()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand Identity Card
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

    # Navigation Menu
    st.markdown("##### Navigation")
    page = st.radio(
        label="Navigation Menu",
        options=["Live Camera", "Video Analysis", "Analytics", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Engine Status
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
        st.caption("Place trained weights at models/best.pt to activate.")

    st.markdown("---")

    # Model Configuration
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

    # Session stats
    st.markdown("##### Session Telemetry")
    summary = get_session_summary()
    st.metric("Total Detections", summary["total_detections"])
    st.metric("Frames Analyzed", summary["frames_processed"])

    if st.button("Reset Telemetry", use_container_width=True, key="sidebar_reset"):
        reset_session()
        st.rerun()

    st.markdown("---")
    st.caption("EcoSort-AI v1.0 · YOLO26 Core")

# ── Model not found notification ──────────────────────────────────────────
if not WEIGHTS_PATH.exists():
    st.warning(
        "Model weights not detected at `models/best.pt`. "
        "Detection pipeline is offline. You can still inspect the platform interface and test modules.",
    )

# ── Load model ─────────────────────────────────────────────────────────────
model = load_model()

# ── Route to selected page ─────────────────────────────────────────────────
if page in ("Live Camera", "🎥 Live Camera"):
    render_camera_page(model, conf=conf, iou=iou)
elif page in ("Video Analysis", "Upload Video", "📁 Upload Video"):
    render_video_page(model, conf=conf, iou=iou)
elif page in ("Analytics", "📊 Analytics", "Analytics Dashboard"):
    render_analytics_page()
else:
    render_about_page()
