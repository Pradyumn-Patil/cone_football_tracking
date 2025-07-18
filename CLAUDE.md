# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the BoT-SORT Tracking System
```bash
# Basic usage with BoT-SORT
python botsort_simple.py input/videos/video.mov -m input/models/model.onnx

# With preview and custom confidence
python botsort_simple.py input/videos/video.mov -m input/models/model.onnx -p -c 0.25

# With custom output path
python botsort_simple.py input/videos/video.mov -m input/models/model.onnx -o output/tracked.mp4
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

This is a **BoT-SORT tracking system** for football and cone detection and tracking.

### Data Flow
1. **botsort_simple.py**: Complete pipeline using custom ONNX model + BoT-SORT tracking
2. **Custom ONNX model**: Detects football (class 1) and cones (class 0) 
3. **BoT-SORT tracker**: Maintains consistent IDs across frames with re-identification
4. **Visualization**: Draws bounding boxes, IDs, and trails

### Key Implementation Details

**Detection System**:
- Uses custom ONNX model at `input/models/model.onnx`
- Detects only 2 classes: cone (0) and football (1)
- Input size: 960x960 pixels

**Tracking Algorithm (BoT-SORT)**:
- Built-in Ultralytics BoT-SORT with default parameters
- Maintains track IDs across frames
- Handles occlusions and re-identification
- Default configuration from `botsort.yaml`

**Visualization**:
- Unique colors for each track ID
- Trail visualization showing object paths
- Real-time bounding boxes with IDs

### File Organization
```
input/
  videos/     # Place input videos here
  models/     # Place model.onnx here
output/       # Processed videos saved here
```