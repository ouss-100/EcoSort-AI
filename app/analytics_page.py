import json
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from app.analytics import get_session_summary, init_session_state, reset_session
from app.ui_components import SVG_ICONS, render_header, render_metric_row
from model.labels import CLASSES


def render_analytics_page() -> None:
    init_session_state()
    render_header(
        title="Analytics & Telemetry Dashboard",
        subtitle="Aggregate detection statistics, latency benchmarks, and dataset exports",
    )

    summary = get_session_summary()
    class_counts: dict = summary.get("class_counts", {})
    times: list = st.session_state.get("inference_times", [])

    render_metric_row({
        "Total Detections": summary["total_detections"],
        "Frames Analyzed": summary["frames_processed"],
        "Mean Latency": f"{summary['avg_inference_ms']} ms",
        "Unique Categories": len([c for c, v in class_counts.items() if v > 0]),
    })

    st.markdown("---")

    if summary["total_detections"] == 0:
        st.markdown(
            f"""
            <div style="
                border: 2px dashed #1f2937;
                border-radius: 12px;
                padding: 3rem 1.5rem;
                text-align: center;
                background: #111827;
                color: #9ca3af;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.75rem;
            ">
                <span style="opacity: 0.7;">{SVG_ICONS['chart']}</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #f9fafb;">No telemetry data recorded in current session</span>
                <span style="font-size: 0.88rem; color: #6b7280; max-width: 480px;">
                    Execute live camera detections or analyze video footage to populate aggregate telemetry, category distributions, and latency metrics.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Category Distribution")
        _render_class_pie(class_counts)
        st.markdown("#### Classification Breakdown")
        _render_class_table(class_counts, summary["total_detections"])

    with col_right:
        st.markdown("#### Inference Latency Timeline")
        _render_inference_timeline(times)
        st.markdown("#### Category Frequency Comparison")
        _render_class_bar(class_counts)

    st.markdown("---")

    st.markdown("#### Export Telemetry Data")
    _render_export(summary)

    st.markdown("---")
    if st.button("Purge Session Telemetry", type="secondary", use_container_width=True):
        reset_session()
        st.success("Session telemetry cleared successfully.")
        st.rerun()


def _render_class_table(class_counts: dict, total: int) -> None:
    rows = []
    for cls in CLASSES:
        count = class_counts.get(cls, 0)
        pct = 100 * count / total if total > 0 else 0.0
        rows.append({
            "Category": cls.capitalize(),
            "Count": count,
            "Share (%)": f"{pct:.1f}%",
        })
    df = pd.DataFrame(rows).sort_values("Count", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_class_pie(class_counts: dict) -> None:
    data = {
        cls.capitalize(): class_counts.get(cls, 0)
        for cls in CLASSES
        if class_counts.get(cls, 0) > 0
    }
    if not data:
        st.info("No distribution data available.")
        return
    df = pd.DataFrame.from_dict(data, orient="index", columns=["Detections"])
    st.bar_chart(df, color="#10b981", horizontal=True)


def _render_class_bar(class_counts: dict) -> None:
    data = {
        cls.capitalize(): class_counts.get(cls, 0)
        for cls in CLASSES
    }
    df = pd.DataFrame.from_dict(data, orient="index", columns=["Detections"])
    df = df[df["Detections"] > 0]
    if df.empty:
        st.info("No category data available.")
        return
    st.bar_chart(df, color="#10b981")


def _render_inference_timeline(times: list) -> None:
    if not times:
        st.info("No inference timing records logged.")
        return
    df = pd.DataFrame({"Frame Index": range(1, len(times) + 1), "Latency (ms)": times})
    df = df.set_index("Frame Index")
    st.line_chart(df, color="#10b981")

    col1, col2, col3 = st.columns(3)
    col1.metric("Minimum", f"{min(times):.1f} ms")
    col2.metric("Mean", f"{sum(times)/len(times):.1f} ms")
    col3.metric("Maximum", f"{max(times):.1f} ms")


def _render_export(summary: dict) -> None:
    col_json, col_csv = st.columns(2)

    report = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_detections": summary["total_detections"],
        "frames_processed": summary["frames_processed"],
        "avg_inference_ms": summary["avg_inference_ms"],
        "class_counts": summary["class_counts"],
    }
    json_str = json.dumps(report, indent=2)
    with col_json:
        st.download_button(
            label="Download JSON Telemetry",
            data=json_str,
            file_name=f"ecosort_telemetry_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    rows = [
        {"class": cls, "count": summary["class_counts"].get(cls, 0)}
        for cls in CLASSES
    ]
    csv_str = pd.DataFrame(rows).to_csv(index=False)
    with col_csv:
        st.download_button(
            label="Download CSV Dataset",
            data=csv_str,
            file_name=f"ecosort_classes_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
