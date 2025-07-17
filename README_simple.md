# Simple Object Detection for Football Videos

This is a simplified version that only performs object detection without tracking or speed visualization.

## Installation

```bash
# Create and activate conda environment
conda create -n yolo_simple python=3.9 -y
conda activate yolo_simple

# Install requirements
pip install -r requirements_simple.txt
```

## Usage

### Basic Usage (Auto-downloads YOLOv8n model)
```bash
python simple_detector.py input/videos/video.mov
```

### With Your ONNX Model
```bash
python simple_detector.py input/videos/video.mov -m input/models/model.onnx
```

### With Preview and Custom Output
```bash
python simple_detector.py input/videos/video.mov -o output/detected.mp4 -p -c 0.6
```

## Command Line Options

- `video`: Path to input video (required)
- `-m, --model`: Path to YOLO model (.pt or .onnx format)
- `-o, --output`: Path to output video (default: input_name_detected.mp4)
- `-c, --confidence`: Detection confidence threshold (default: 0.5)
- `-p, --preview`: Show live preview window

## Features

- Simple bounding box detection
- No tracking (each frame processed independently)
- Supports both .pt and .onnx models
- Color-coded boxes (green for ball, red for cone, blue for others)
- Shows confidence scores
- Progress indicator with ETA

## Model Support

- **Default**: Uses YOLOv8n (downloads automatically)
- **Custom .pt models**: Any Ultralytics YOLO model
- **ONNX models**: Your trained ONNX models

## Performance

This simplified version is much faster because:
- No tracking overhead
- Uses optimized Ultralytics implementation
- Minimal post-processing
- No path history or speed calculations

## Output

The script produces a video with:
- Bounding boxes around detected objects
- Class labels and confidence scores
- Color-coded boxes based on object type