# EcoSort-AI

> High-throughput computer vision platform for real-time waste detection and automated material sorting, powered by fine-tuned YOLO26 and Streamlit.

Classifies refuse into 6 core industrial streams: **Biodegradable**, **Cardboard**, **Glass**, **Metal**, **Paper**, and **Plastic**.

Benchmark Dataset: [Garbage Classification 3 — Roboflow Universe](https://universe.roboflow.com/waste-classification-jl6nz/garbage-classification-3-ymuqp/dataset/3) (~15,000 labeled samples).

---

## Quick Start

### 1. Environment Setup

```bash
pip install -e .
```

Or using `uv`:
```bash
uv sync
```

### 2. Model Weights Verification

Place your trained `best.pt` file inside the `models/` directory:

```
EcoSort-AI/
└── models/
    └── best.pt   <-- verified weights
```

### 3. Launch Platform Interface

```bash
streamlit run app.py
# or via CLI
python main.py app
```

### 4. CLI Execution

```bash
python main.py status                          # Verify environment and model readiness
python main.py camera --camera-index 0         # High-FPS desktop OpenCV loop
python main.py video input.mp4 --skip 2        # Batch video processing and annotated export
```

---

## Platform Modules

### 1. Live Camera Detection
- High-definition viewport capture via browser camera interface
- Low-latency YOLO26 inference with non-maximum suppression (NMS)
- Spatial bounding boxes with category confidence scores
- Dynamic category distribution breakdown and session-level telemetry

### 2. Video Stream Analysis
- Multi-format ingestion: MP4, AVI, MOV, and MKV (up to 500 MB)
- Configurable inference stride (1–10) and resolution presets (Original, 720p HD, 480p SD)
- Stream-buffered export pipeline protecting system memory
- Side-by-side comparative inspection between raw footage and annotated stream
- Downloadable annotated footage with full chronological telemetry

### 3. Analytics & Telemetry Dashboard
- Aggregate classification metrics: total detections, analyzed frames, mean latency
- Inference latency distribution and chronological timeline tracking
- Cross-category comparative histograms
- Standardized schema export to JSON telemetry reports and CSV dataset files

### 4. Platform Specifications & Documentation
- Architectural overview and dataset specifications
- Technical parameter guide for confidence gating and IoU suppression
- Material stream color coding and industrial compliance guidelines

---

## Material Classes & Specifications

| ID | Class Category | Accent Color | Description |
|:---|:---|:---|:---|
| 0 | Biodegradable | `#27ae60` Green | Organic and compostable matter |
| 1 | Cardboard | `#e67e22` Orange | Corrugated containers and packaging boxes |
| 2 | Glass | `#3498db` Blue | Bottles, jars, and transparent silica containers |
| 3 | Metal | `#95a5a6` Silver | Aluminum cans, tin, scrap containers |
| 4 | Paper | `#f1c40f` Yellow | Office paper, newsprint, and clean fiber |
| 5 | Plastic | `#e74c3c` Red | Polymers (PET, HDPE, LDPE, PP, PS) |

---

## Inference Hyperparameters

| Parameter | Default | Operational Impact |
|:---|:---|:---|
| `conf` | `0.35` | Minimum classification confidence required to register detection |
| `iou` | `0.45` | Non-Maximum Suppression threshold to filter spatial duplicates |
| `imgsz` | `640` | Square input resolution for network tensor processing |

---

## Test Verification

```bash
python -m pytest tests/ -v
```

92 passing unit and integration tests covering inference pipelines, analytics, centroid tracking, video I/O, rendering overlays, and cache management.
