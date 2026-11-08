"""
EcoSort-AI — CLI entry point.

Usage
-----
# Launch the Streamlit web app (recommended)
python main.py app

# Run real-time camera detection in an OpenCV window
python main.py camera [--camera-index 0] [--conf 0.35] [--iou 0.45]

# Process a video file from the command line
python main.py video INPUT.mp4 [--output OUTPUT.mp4] [--conf 0.35] [--iou 0.45] [--skip 1]

# Print model status
python main.py status
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ecosort")

# ── Project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DEFAULT_WEIGHTS, CONF_THRESHOLD, IOU_THRESHOLD


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_app(args) -> None:
    """Launch the Streamlit app."""
    logger.info("Launching EcoSort-AI Streamlit app…")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")],
        check=False,
    )


def cmd_status(args) -> None:
    """Print model and environment status."""
    from model.model import is_model_available

    print("\n[EcoSort-AI System Status]")
    print("=" * 40)

    # Model
    if is_model_available():
        print(f"  [OK]   Model:   {DEFAULT_WEIGHTS} (found)")
    else:
        print(f"  [FAIL] Model:   {DEFAULT_WEIGHTS} (NOT FOUND)")
        print("         -> Copy your trained best.pt into the models/ folder.")

    # Python
    print(f"  [INFO] Python:  {sys.version.split()[0]}")

    # Packages
    for pkg in ["ultralytics", "streamlit", "cv2", "torch", "numpy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  [OK]   {pkg:<14} {ver}")
        except ImportError:
            print(f"  [WARN] {pkg:<14} NOT INSTALLED")

    print()


def cmd_camera(args) -> None:
    """Run real-time camera detection in an OpenCV window."""
    from model.model import get_model
    from src.camera import realtime_detection_loop

    logger.info("Loading model from: %s", DEFAULT_WEIGHTS)
    model = get_model()
    if model is None:
        logger.error("Model not found. Place best.pt in models/ and retry.")
        sys.exit(1)

    logger.info("Starting camera detection on index %d …", args.camera_index)
    realtime_detection_loop(
        model,
        camera_index=args.camera_index,
        conf=args.conf,
        iou=args.iou,
    )


def cmd_video(args) -> None:
    """Process a video file and save annotated output."""
    from model.model import get_model
    from src.pipline import process_video_file

    input_path = Path(args.video)
    if not input_path.exists():
        logger.error("Input video not found: %s", input_path)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_stem(
        input_path.stem + "_processed"
    )

    logger.info("Loading model…")
    model = get_model()
    if model is None:
        logger.error("Model not found. Place best.pt in models/ and retry.")
        sys.exit(1)

    logger.info("Processing: %s → %s", input_path, output_path)

    def progress(current, total):
        pct = int(100 * current / max(total, 1))
        print(f"\r  [RUN]  {pct:3d}%  [{current}/{total} frames]", end="", flush=True)

    report = process_video_file(
        input_path=input_path,
        output_path=output_path,
        model=model,
        conf=args.conf,
        iou=args.iou,
        frame_skip=args.skip,
        annotate=True,
        progress_callback=progress,
    )

    print()  # newline after progress
    print("\n[Processing Summary]")
    print("=" * 40)
    for k, v in report.to_dict().items():
        if k != "class_counts":
            print(f"  {k:<25} {v}")
    print("  Class counts:")
    for cls, cnt in sorted(report.class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<20} {cnt}")
    print(f"\n  [DONE] Saved to: {output_path}\n")


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecosort",
        description="EcoSort-AI — Waste Detection & Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # app
    sub.add_parser("app", help="Launch the Streamlit web app")

    # status
    sub.add_parser("status", help="Print model and environment status")

    # camera
    cam = sub.add_parser("camera", help="Real-time detection via webcam (OpenCV window)")
    cam.add_argument("--camera-index", type=int, default=0, metavar="N",
                     help="Webcam index (default: 0)")
    cam.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                     help=f"Confidence threshold (default: {CONF_THRESHOLD})")
    cam.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                     help=f"IoU threshold (default: {IOU_THRESHOLD})")

    # video
    vid = sub.add_parser("video", help="Process a video file and save annotated output")
    vid.add_argument("video", metavar="INPUT", help="Path to the input video file")
    vid.add_argument("--output", "-o", metavar="OUTPUT",
                     help="Output path (default: <input>_processed.mp4)")
    vid.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                     help=f"Confidence threshold (default: {CONF_THRESHOLD})")
    vid.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                     help=f"IoU threshold (default: {IOU_THRESHOLD})")
    vid.add_argument("--skip", type=int, default=1, metavar="N",
                     help="Process every N frames (default: 1)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "app":    cmd_app,
        "status": cmd_status,
        "camera": cmd_camera,
        "video":  cmd_video,
    }

    if args.command in handlers:
        handlers[args.command](args)
    else:
        # Default: launch the app
        cmd_app(args)


if __name__ == "__main__":
    main()
