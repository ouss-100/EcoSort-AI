"""Detection table component — self-contained."""
import pandas as pd
import streamlit as st


def render_detection_table(result) -> None:
    if not result.detections:
        st.info("No detections identified in this frame.")
        return
    rows = []
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        rows.append({
            "Category":    det.class_name.capitalize(),
            "Confidence":  f"{det.confidence * 100:.1f}%",
            "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
