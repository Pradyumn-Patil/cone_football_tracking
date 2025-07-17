# Football and Cone Tracking with Speed-Colored Path Visualization

This project implements a robust tracking system for football games that detects and tracks footballs and cones throughout a video. The key feature is visualizing the football's path with colors indicating its real-time speed.

## Features

- **Object Detection**: Uses YOLO ONNX model to detect footballs (class 0) and cones (class 1)
- **Robust Tracking**: Implements ByteTrack algorithm to handle intermittent detections and false positives
- **Speed Visualization**: Football path is colored based on speed (green=slow, yellow=medium, red=fast)
- **Persistent IDs**: Each tracked object maintains the same ID throughout the video
- **Noise Filtering**: Automatically filters out short-lived false positive detections

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py video.mp4 model.onnx
```

### Advanced Options
```bash
python main.py video.mp4 model.onnx -o output_video.mp4 -c 0.6 -p
```

### Command Line Arguments
- `video`: Path to input video file (required)
- `model`: Path to YOLO ONNX model file (required)
- `-o, --output`: Path to output video file (default: input_name_tracked.mp4)
- `-c, --confidence`: Detection confidence threshold (default: 0.5)
- `-p, --preview`: Show live preview during processing

## How It Works

1. **Detection**: Each frame is processed through the YOLO model to detect objects
2. **Tracking**: ByteTrack associates detections across frames, maintaining object identities
3. **Speed Calculation**: For footballs, the system calculates speed based on position changes
4. **Visualization**: 
   - All objects get bounding boxes with unique IDs
   - Football paths are drawn with speed-based colors
   - A speed legend is added to the video

## Handling Detection Challenges

The system is designed to handle common detection issues:

- **Missed Detections**: Tracks are maintained for up to 30 frames even without detections
- **False Positives**: Tracks appearing for less than 10 frames are automatically filtered out
- **Occlusions**: ByteTrack's motion prediction helps maintain tracks through brief occlusions

## Speed Color Mapping

The football path color represents speed:
- 🟢 **Green**: Low speed (< 5 pixels/frame)
- 🟡 **Yellow/Orange**: Medium speed (5-25 pixels/frame)
- 🔴 **Red**: High speed (> 25 pixels/frame)

## Output

The system produces an annotated video with:
- Bounding boxes around all tracked objects
- Unique IDs displayed above each object
- Colored path trail for the football showing speed variations
- Speed color legend in the top-right corner

## Project Structure

- `main.py`: Main pipeline orchestrating the entire process
- `detector.py`: YOLO ONNX model wrapper for object detection
- `tracker.py`: ByteTrack implementation for robust tracking
- `visualizer.py`: Handles all visualization including speed-colored paths
- `utils.py`: Helper functions for speed calculation and file handling
- `requirements.txt`: Project dependencies

## Requirements

- Python 3.8+
- ONNX model file (model.onnx) with football and cone classes
- Input video file (video.mp4)

## Troubleshooting

### "Model file not found"
Ensure the model.onnx file is in the correct location and the path is correct.

### Low FPS during processing
- Reduce video resolution before processing
- Adjust confidence threshold to reduce detections
- Disable preview mode (-p flag)

### Tracking issues
- Adjust confidence threshold (-c flag)
- Modify tracking parameters in tracker.py (max_time_lost, min_hits)