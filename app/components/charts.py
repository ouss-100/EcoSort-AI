"""Chart components — self-contained."""
from collections import Counter
from typing import List

import pandas as pd
import streamlit as st

from core.detection.labels import CLASSES


def render_class_chart(detections_list: List) -> None:
    if not detections_list:
        st.info("No detections to chart.")
        return
    counts = Counter(det.class_name for det in detections_list)
    chart_data = {cls.capitalize(): counts.get(cls, 0) for cls in CLASSES}
    df = pd.DataFrame.from_dict(chart_data, orient="index", columns=["Detections"])
    df = df[df["Detections"] > 0]
    if df.empty:
        st.info("No active detections to visualize.")
        return
    st.bar_chart(df, horizontal=True, color="#10b981")
