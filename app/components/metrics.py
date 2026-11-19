"""Metric row component — self-contained."""
import re
import streamlit as st
from typing import Dict


def clean_label(text: str) -> str:
    cleaned = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff]', '', str(text))
    return cleaned.strip()


def render_metric_row(metrics: Dict[str, object]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label=clean_label(label), value=clean_label(str(value)))
