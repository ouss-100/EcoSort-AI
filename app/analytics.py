from collections import defaultdict
from typing import Dict
import streamlit as st
from model.labels import CLASSES


def init_session_state() -> None:
    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0
    if "frames_processed" not in st.session_state:
        st.session_state.frames_processed = 0
    if "class_counts" not in st.session_state:
        st.session_state.class_counts = defaultdict(int)
    if "inference_times" not in st.session_state:
        st.session_state.inference_times = []


def update_stats(result) -> None:
    init_session_state()
    st.session_state.frames_processed += 1
    st.session_state.total_detections += result.num_detections
    for det in result.detections:
        st.session_state.class_counts[det.class_name] += 1
    if result.inference_ms > 0:
        st.session_state.inference_times.append(result.inference_ms)


def get_session_summary() -> Dict:
    init_session_state()
    times = st.session_state.inference_times
    avg_ms = round(sum(times) / len(times), 1) if times else 0.0
    return {
        "total_detections": st.session_state.total_detections,
        "frames_processed": st.session_state.frames_processed,
        "class_counts": dict(st.session_state.class_counts),
        "avg_inference_ms": avg_ms,
    }


def reset_session() -> None:
    st.session_state.total_detections = 0
    st.session_state.frames_processed = 0
    st.session_state.class_counts = defaultdict(int)
    st.session_state.inference_times = []
