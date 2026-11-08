import atexit
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import pandas as pd
import streamlit as st
from app.analytics import init_session_state, update_stats
from model.inference import Detection, draw_detections, run_inference
from app.ui_components import (
    render_class_chart,
    render_header,
    render_metric_row,
)

QUALITY_PRESETS: Dict[str, Optional[Tuple[int, int]]] = {
    "Original": None,
    "720p (HD)": (1280, 720),
    "480p (SD)": (854, 480),
}


def render_video_page(model, conf: float, iou: float) -> None:
    init_session_state()

    render_header(
        title="Video Analysis",
        subtitle="Automated batch computer vision pipeline with annotated stream export",
    )

    if model is None:
        st.warning(
            "Model weights not loaded — detection pipeline will be skipped during processing.",
        )

    st.markdown("#### Input Footage")
    uploaded_file = st.file_uploader(
        "Upload Video Source",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported containers: MP4, AVI, MOV, MKV. Maximum file size: 500 MB.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        _cleanup_prev_temp()
        return

    file_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("_video_temp_key") != file_key:
        _cleanup_prev_temp()
        uploaded_file.seek(0)
        tfile = _save_upload_to_temp(uploaded_file)
        st.session_state["_video_temp_path"] = tfile
        st.session_state["_video_temp_key"] = file_key
    tfile = st.session_state["_video_temp_path"]

    try:
        cap = cv2.VideoCapture(tfile)
        if not cap.isOpened():
            st.error("Could not open the uploaded video. Please verify the codec and format.")
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration   = total_frames / fps if fps > 0 else 0
        cap.release()
    except Exception as exc:
        st.error(f"Error reading video metadata: {exc}")
        return

    _render_video_info(uploaded_file.name, duration, fps, width, height, total_frames)

    st.markdown("---")

    st.markdown("#### Processing Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        frame_skip = st.slider(
            "Sampling Stride (Every N Frames)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Higher values accelerate inference throughput but reduce detection frequency.",
        )
    with col2:
        quality_choice = st.selectbox(
            "Target Output Resolution",
            options=list(QUALITY_PRESETS.keys()),
            index=0,
        )
    with col3:
        show_annotations = st.toggle("Overlay Bounding Boxes", value=True)

    st.markdown("---")

    process_btn = st.button(
        "Start Video Processing",
        type="primary",
        use_container_width=True,
        disabled=(model is None),
    )

    if model is None:
        st.caption("Place best.pt in models/ directory to activate detection engine.")

    if process_btn:
        output_path, all_detections, stats = _process_video(
            tfile,
            model,
            conf,
            iou,
            fps,
            width,
            height,
            total_frames,
            frame_skip,
            QUALITY_PRESETS[quality_choice],
            show_annotations,
        )
        if output_path:
            _render_results(tfile, output_path, all_detections, stats)

def _cleanup_prev_temp() -> None:
    prev = st.session_state.pop("_video_temp_path", None)
    st.session_state.pop("_video_temp_key", None)
    if prev:
        try:
            Path(prev).unlink(missing_ok=True)
        except OSError:
            pass


def _save_upload_to_temp(uploaded_file) -> str:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    tfile.close()
    atexit.register(lambda p=tfile.name: Path(p).unlink(missing_ok=True))
    return tfile.name


def _render_video_info(
    name: str, duration: float, fps: float, width: int, height: int, total_frames: int
) -> None:
    st.markdown("#### Source Metadata")
    render_metric_row(
        {
            "File Name": name,
            "Duration": f"{duration:.1f}s",
            "Framerate": f"{fps:.1f} FPS",
            "Resolution": f"{width}×{height}",
            "Frame Count": total_frames,
        }
    )


def _compute_output_size(
    orig_w: int, orig_h: int, preset: Optional[Tuple[int, int]]
) -> Tuple[int, int]:
    if preset is None:
        return orig_w, orig_h
    target_w, target_h = preset
    ratio = min(target_w / orig_w, target_h / orig_h)
    return int(orig_w * ratio), int(orig_h * ratio)


def _process_video(
    input_path: str,
    model,
    conf: float,
    iou: float,
    fps: float,
    orig_width: int,
    orig_height: int,
    total_frames: int,
    frame_skip: int,
    quality_preset: Optional[Tuple[int, int]],
    show_annotations: bool,
) -> Tuple[Optional[str], List[Detection], Dict]:
    out_width, out_height = _compute_output_size(orig_width, orig_height, quality_preset)
    out_path = str(Path(input_path).with_stem(Path(input_path).stem + "_processed").with_suffix(".mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height))

    cap = cv2.VideoCapture(input_path)

    progress_bar = st.progress(0.0, text="Initializing processing pipeline...")
    status_text  = st.empty()

    all_detections: List[Detection] = []
    inference_times: List[float] = []
    frame_idx = 0
    written   = 0
    t_start   = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                progress = min(frame_idx / max(total_frames, 1), 1.0)
                progress_bar.progress(progress, text=f"Processing frame {frame_idx} of {total_frames}...")

            # Resize if needed
            if (out_width, out_height) != (orig_width, orig_height):
                frame = cv2.resize(frame, (out_width, out_height))

            if (frame_idx - 1) % frame_skip == 0 and model is not None:
                result = run_inference(model, frame, conf=conf, iou=iou)
                if not result.error:
                    all_detections.extend(result.detections)
                    inference_times.append(result.inference_ms)
                    update_stats(result)
                    if show_annotations:
                        frame = draw_detections(frame, result)

            writer.write(frame)
            written += 1

    except Exception as exc:
        st.error(f"Error during video processing: {exc}")
        return None, [], {}
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t_start
    avg_inf = round(sum(inference_times) / len(inference_times), 1) if inference_times else 0.0
    avg_fps = round(written / elapsed, 1) if elapsed > 0 else 0.0

    progress_bar.progress(1.0, text="Processing completed successfully")
    status_text.success(f"Processed {written} frames in {elapsed:.1f}s ({avg_fps} FPS average throughput)")

    stats = {
        "frames_written": written,
        "total_time_s": round(elapsed, 1),
        "avg_fps": avg_fps,
        "avg_inference_ms": avg_inf,
        "total_detections": len(all_detections),
    }
    return out_path, all_detections, stats


def _render_results(
    original_path: str,
    processed_path: str,
    all_detections: List[Detection],
    stats: Dict,
) -> None:
    st.markdown("---")
    st.markdown("#### Output Inspection")

    tab_annotated, tab_comparison = st.tabs(["Annotated Stream", "Side-by-Side Comparison"])

    with tab_annotated:
        st.video(processed_path)

    with tab_comparison:
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.markdown("**Source Footage**")
            st.video(original_path)
        with col_proc:
            st.markdown("**Annotated Inference Stream**")
            st.video(processed_path)

    try:
        file_size_mb = Path(processed_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 200:
            st.warning(
                f"Processed output file size is {file_size_mb:.0f} MB. "
                "Download may take some time depending on network throughput.",
            )

        def _video_stream(path: str, chunk: int = 1 << 20):
            with open(path, "rb") as fh:
                while True:
                    data = fh.read(chunk)
                    if not data:
                        break
                    yield data

        st.download_button(
            label="Download Processed Video (MP4)",
            data=b"".join(_video_stream(processed_path)),
            file_name="ecosort_processed.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Could not prepare download: {exc}")

    st.markdown("#### Performance Metrics")
    render_metric_row(
        {
            "Total Time": f"{stats['total_time_s']}s",
            "Throughput": f"{stats['avg_fps']} FPS",
            "Latency": f"{stats['avg_inference_ms']} ms",
            "Detections": stats["total_detections"],
        }
    )

    st.markdown("#### Category Distribution")
    if all_detections:
        counts = Counter(det.class_name for det in all_detections)
        rows = [
            {
                "Category": cls.capitalize(),
                "Count": counts.get(cls, 0),
                "Share": f"{100 * counts.get(cls, 0) / max(len(all_detections), 1):.1f}%",
            }
            for cls in sorted(counts, key=counts.get, reverse=True)
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("##### Category Breakdown")
        render_class_chart(all_detections)
    else:
        st.info("No items detected in processed video.")
