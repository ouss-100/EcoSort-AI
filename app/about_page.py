import html
import streamlit as st
from app.ui_components import SVG_ICONS, render_header
from model.labels import CLASS_COLORS_HEX, CLASSES
from model.model import is_model_available


def render_about_page() -> None:
    render_header(
        title="Platform Specifications & Architecture",
        subtitle="Technical documentation, inference pipeline, and category specifications",
    )

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("""
### Architecture Overview

**EcoSort-AI** is a high-performance computer vision platform designed for real-time waste sorting and material classification. Built on a fine-tuned **YOLO26** object detection network, the system analyzes visual feeds to localize and classify refuse into actionable material streams.

The core model is trained on the [Garbage Classification 3](https://universe.roboflow.com/waste-classification-jl6nz/garbage-classification-3-ymuqp/dataset/3) benchmark dataset (~15,000 labeled industrial and consumer waste images), achieving sub-30ms inference latencies on standard hardware.

---

### Operational Modes

| Operational Mode | Purpose | Input Specifications | Output Format |
| :--- | :--- | :--- | :--- |
| **Live Viewport** | Real-time edge classification | Camera device snapshot | Annotated viewport & bounding coordinates |
| **Video Stream** | High-throughput batch auditing | MP4, AVI, MOV, MKV (<500MB) | Re-encoded video & chronological telemetry |
| **Telemetry Dashboard** | Performance & compliance auditing | Session state data | JSON schema export & CSV breakdown |

---

### Hyperparameter Specifications

The inference engine exposes two primary tuning parameters in the configuration panel:

- **Confidence Threshold (`conf`)**  
  *Default: 0.35* — Lower values maximize recall at the cost of potential false positives. Higher values ensure strict precision for robotic pick-and-place automation.
- **IoU Suppression Threshold (`iou`)**  
  *Default: 0.45* — Controls Non-Maximum Suppression (NMS) overlap gating to resolve duplicate bounding boxes around identical items.

---

### Operational Best Practices

1. **Illumination Consistency**: Ensure balanced diffuse lighting to reduce specular glare on metallic and transparent glass surfaces.
2. **Sampling Stride Optimization**: When processing high-framerate footage (60+ FPS), configure a frame stride of 2–5 to maximize throughput without missing transient items.
3. **Bandwidth Efficiency**: Select 480p SD or 720p HD output presets for remote edge deployments to conserve network bandwidth during export.
""")

    with col2:
        st.markdown("### Material Categories")
        for cls in CLASSES:
            color = CLASS_COLORS_HEX.get(cls, "#9ca3af")
            safe_color = html.escape(color)
            safe_cls = html.escape(cls.capitalize())
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0.6rem 0.85rem;
                    margin-bottom: 0.45rem;
                    border: 1px solid #1f2937;
                    border-left: 4px solid {safe_color};
                    background: #111827;
                    border-radius: 6px;
                ">
                    <span style="font-weight: 600; color: #f3f4f6; font-size: 0.9rem;">{safe_cls}</span>
                    <span style="
                        font-family: monospace;
                        font-size: 0.75rem;
                        color: #9ca3af;
                        background: #1f2937;
                        padding: 0.15rem 0.45rem;
                        border-radius: 4px;
                        border: 1px solid #374151;
                    ">{safe_color}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Engine Verification")
        if is_model_available():
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    background: rgba(16, 185, 129, 0.12);
                    border: 1px solid rgba(16, 185, 129, 0.35);
                    padding: 0.65rem 0.85rem;
                    border-radius: 8px;
                    color: #34d399;
                    font-size: 0.85rem;
                    font-weight: 600;
                ">
                    <span>{SVG_ICONS['check']}</span>
                    <span>Weights Verified · models/best.pt</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    background: rgba(245, 158, 11, 0.12);
                    border: 1px solid rgba(245, 158, 11, 0.35);
                    padding: 0.65rem 0.85rem;
                    border-radius: 8px;
                    color: #fbbf24;
                    font-size: 0.85rem;
                    font-weight: 600;
                ">
                    <span>{SVG_ICONS['alert']}</span>
                    <span>Missing Weights · models/best.pt</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Documentation & Repositories")
        st.markdown("""
- [Training Notebook & Benchmarks](https://github.com/ultralytics/ultralytics)
- [Roboflow Garbage Classification 3 Dataset](https://universe.roboflow.com/waste-classification-jl6nz/garbage-classification-3-ymuqp/dataset/3)
- [Ultralytics YOLO Architecture Reference](https://github.com/ultralytics/ultralytics)
        """)

        st.markdown("---")
        st.caption("EcoSort-AI Platform · Build v1.0.4 · Enterprise Vision Core")
