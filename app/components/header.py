"""Page header component — self-contained, no dependency on ui_components."""
import html
import re
import streamlit as st
from app.components.icons import SVG_ICONS


def clean_label(text: str) -> str:
    """Strip emoji / surrogate characters from a string."""
    cleaned = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff]', '', str(text))
    return cleaned.strip()


def render_header(
    title: str = "EcoSort-AI",
    subtitle: str = "AI-Powered Waste Detection & Classification Platform",
) -> None:
    safe_title    = html.escape(clean_label(title))
    safe_subtitle = html.escape(clean_label(subtitle))
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
                        font-size: 1.55rem; font-weight: 800; color: #f9fafb;
                        margin: 0; letter-spacing: -0.025em;
                    ">{safe_title}</h1>
                </div>
                <p style="
                    color: #9ca3af; font-size: 0.88rem;
                    margin: 0.25rem 0 0 0; font-weight: 400;
                ">{safe_subtitle}</p>
            </div>
            <div style="
                display: flex; align-items: center; gap: 0.45rem;
                background: rgba(16,185,129,0.1); color: #34d399;
                padding: 0.35rem 0.75rem; border-radius: 9999px;
                border: 1px solid rgba(16,185,129,0.3);
                font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em;
            ">
                <span style="width:7px;height:7px;border-radius:50%;
                             background:#10b981;display:inline-block;
                             box-shadow:0 0 8px #10b981;"></span>
                YOLO26 ENGINE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
