import html
import re
from typing import Dict, List
import pandas as pd
import streamlit as st
from model.labels import CLASSES
from collections import Counter

CLASS_EMOJI: Dict[str, str] = {
    "biodegradable": "",
    "cardboard":     "",
    "glass":         "",
    "metal":         "",
    "paper":         "",
    "plastic":       "",
}

SVG_ICONS: Dict[str, str] = {
    "leaf": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>""",
    "camera": """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>""",
    "video": """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="14" x="2" y="5" rx="2"/><polygon points="12 10 16 12 12 14 12 10"/></svg>""",
    "chart": """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>""",
    "upload": """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>""",
    "check": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>""",
    "alert": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>""",
}


def clean_label(text: str) -> str:
    cleaned = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff]', '', str(text))
    return cleaned.strip()


def render_header(
    title: str = "EcoSort-AI",
    subtitle: str = "AI-Powered Waste Detection & Classification Platform"
) -> None:
    clean_title = clean_label(title)
    clean_sub = clean_label(subtitle)
    safe_title = html.escape(clean_title)
    safe_subtitle = html.escape(clean_sub)

    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1.25rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #1f2937;
        ">
            <div>
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <span>{SVG_ICONS["leaf"]}</span>
                    <h1 style="
                        font-size: 1.55rem;
                        font-weight: 800;
                        color: #f9fafb;
                        margin: 0;
                        letter-spacing: -0.025em;
                    ">{safe_title}</h1>
                </div>
                <p style="
                    color: #9ca3af;
                    font-size: 0.88rem;
                    margin: 0.25rem 0 0 0;
                    font-weight: 400;
                ">{safe_subtitle}</p>
            </div>
            <div style="
                display: flex;
                align-items: center;
                gap: 0.45rem;
                background: rgba(16, 185, 129, 0.1);
                color: #34d399;
                padding: 0.35rem 0.75rem;
                border-radius: 9999px;
                border: 1px solid rgba(16, 185, 129, 0.3);
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.04em;
            ">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
                YOLO26 ENGINE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: Dict[str, object]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        cl_label = clean_label(label)
        cl_val = clean_label(str(value))
        col.metric(label=cl_label, value=cl_val)


def render_detection_table(result) -> None:
    if not result.detections:
        st.info("No detections identified in this frame.")
        return

    rows = []
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        rows.append(
            {
                "Category": det.class_name.capitalize(),
                "Confidence": f"{det.confidence * 100:.1f}%",
                "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_class_chart(detections_list: List) -> None:
    if not detections_list:
        st.info("No detections to chart.")
        return

    counts = Counter(det.class_name for det in detections_list)

    chart_data = {
        cls.capitalize(): counts.get(cls, 0)
        for cls in CLASSES
    }
    df = pd.DataFrame.from_dict(chart_data, orient="index", columns=["Detections"])
    df = df[df["Detections"] > 0]

    if df.empty:
        st.info("No active detections to visualize.")
        return

    st.bar_chart(df, horizontal=True, color="#10b981")
