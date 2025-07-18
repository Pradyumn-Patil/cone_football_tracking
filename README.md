# Football & Cone Tracking with BoT-SORT

Simple and effective object tracking using a custom ONNX model with BoT-SORT algorithm.

## Features

- **Custom ONNX Model**: Uses your trained model for football and cone detection
- **BoT-SORT Tracking**: State-of-the-art tracking with ID persistence
- **Real-time Visualization**: Live tracking with trails and bounding boxes

## Installation

```bash
# Create environment
conda create -n football_tracking python=3.9 -y
conda activate football_tracking

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic tracking
python botsort_simple.py input/videos/video.mov -m input/models/model.onnx

# With preview
python botsort_simple.py input/videos/video.mov -m input/models/model.onnx -p

# Custom confidence threshold
python botsort_simple.py input/videos/video.mov -m input/models/model.onnx -c 0.3 -p
```

## How It Works

1. **Detection**: Your ONNX model detects cones (class 0) and football (class 1)
2. **Tracking**: BoT-SORT maintains consistent IDs across frames
3. **Visualization**: Shows bounding boxes, IDs, and movement trails

## Requirements

- Python 3.9+
- Custom ONNX model at `input/models/model.onnx`
- Input videos in `input/videos/`

That's it! Simple, clean, and effective tracking.