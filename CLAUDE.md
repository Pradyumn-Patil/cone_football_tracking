# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Tracking System
```bash
# Basic usage with files in standard locations
python main.py input/videos/video.mov input/models/model.onnx

# With output path and preview
python main.py input/videos/video.mov input/models/model.onnx -o output/tracked_video.mp4 -p

# Adjusting detection confidence (default: 0.5)
python main.py input/videos/video.mov input/models/model.onnx -c 0.6
```

### Environment Setup
```bash
# Using conda (recommended)
conda create -n football_tracking python=3.9 -y
conda activate football_tracking
pip install -r requirements.txt

# Using venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Architecture Overview

This is a **detection-tracking-visualization pipeline** for football and cone tracking with speed-based path coloring.

### Data Flow
1. **main.py** orchestrates: video → frame extraction → detection → tracking → visualization → output video
2. **detector.py** wraps YOLO ONNX model, handling preprocessing and coordinate transformation
3. **tracker.py** implements ByteTrack with two-stage matching (high then low confidence detections)
4. **visualizer.py** draws bounding boxes, IDs, and speed-colored paths (football only)
5. **utils.py** provides validation and helper functions

### Key Implementation Details

**Detection System**:
- Uses ONNX Runtime (not PyTorch/TensorFlow)
- Detects only 2 classes: football (0) and cone (1)
- Dynamic input shape adaptation from model

**Tracking Algorithm (ByteTrack)**:
- Two-stage matching: first high-confidence, then recovers tracks with low-confidence detections
- Track persistence: maintains tracks for 30 frames without detection
- False positive filtering: requires 10 frames minimum to be valid
- Uses scipy's linear_sum_assignment (modified from original lap implementation)

**Speed Visualization**:
- Only football paths are colored by speed
- Uses matplotlib's RdYlGn_r colormap
- Speed thresholds: 5, 15, 25 pixels/frame for green→yellow→red
- Path history limited to 100 points per track

### Critical Parameters

In `tracker.py`:
- `max_time_lost=30`: frames to maintain track without detection
- `min_hits=10`: minimum detections before track is valid
- `track_thresh=0.5`: minimum confidence for primary tracking
- `match_thresh=0.8`: IoU threshold for detection-track matching

In `visualizer.py`:
- `speed_thresholds=(5, 15, 25)`: pixel/frame thresholds for color mapping

### File Organization
```
input/
  videos/     # Place input videos here
  models/     # Place model.onnx here
output/       # Processed videos saved here
```