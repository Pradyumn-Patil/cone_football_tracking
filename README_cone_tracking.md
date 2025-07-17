# Cone Tracking System

This advanced cone tracking system provides unique, persistent IDs for each cone throughout a video, handling camera movements and temporary detection failures.

## Features

### 🎯 Kalman Filter-based Tracking
- Predicts cone positions when temporarily not detected
- Smooths trajectories and handles motion uncertainty
- Adapts to both static cones and moving camera

### 🔄 Dynamic IoU Matching
- No hard-coded distance thresholds
- Automatically scales with cone size (near/far from camera)
- Optimal global assignment using Hungarian algorithm

### 🎨 Visual Features
- Each cone gets a unique color based on its ID
- Solid boxes for confirmed tracks, dashed for tentative
- Optional movement trails
- Real-time tracking statistics
- Football highlighted with thicker green box and trail

### 💪 Robustness
- Handles detection failures for up to 30 frames
- Confidence-based track management
- Filters out false positives (requires 3 detections to confirm)

### ⚽ Football-Aware Tracking (NEW)
- Prevents football from creating false cone tracks
- Maintains football position history
- Validates new cones aren't near recent football positions
- Size-based validation for consistent cone detection

## Installation

```bash
# Install additional dependencies
pip install -r requirements_simple.txt
```

## Usage

### Basic Cone Tracking
```bash
python simple_detector.py input/videos/video.mov -m input/models/model.onnx --imgsz 960 --track-cones
```

### With Strict Validation (Recommended for football scenes)
```bash
python simple_detector.py input/videos/video.mov -m input/models/model.onnx --imgsz 960 --track-cones --strict-validation
```

### With Custom Football Buffer Zone
```bash
python simple_detector.py input/videos/video.mov -m input/models/model.onnx --imgsz 960 --track-cones --strict-validation --football-buffer 75
```

### Full Features with Preview
```bash
python simple_detector.py input/videos/video.mov -m input/models/model.onnx --imgsz 960 --track-cones --show-trails --strict-validation --save-tracks tracking_data.json -p
```

## Command Line Options

- `--track-cones`: Enable cone tracking with unique IDs
- `--show-trails`: Display movement trails for each cone
- `--save-tracks`: Export tracking data to JSON file
- `--strict-validation`: Enable strict cone validation to prevent football misclassification
- `--football-buffer N`: Set minimum distance from football for new cone tracks (default: 50 pixels)

## How It Works

1. **Detection**: YOLO detects all objects in each frame
2. **Classification**: Cones are separated from other objects
3. **Prediction**: Kalman filters predict positions of existing tracks
4. **Matching**: IoU-based Hungarian algorithm matches detections to tracks
5. **Update**: Matched tracks are updated, new tracks created for unmatched detections
6. **Visualization**: Each cone displayed with unique ID and color

## Tracking States

- **Tentative** (dashed box): New track with < 3 detections
- **Confirmed** (solid box): Established track with ≥ 3 detections
- **Lost**: Track not detected but maintained by Kalman prediction
- **Deleted**: Track removed after 30 frames without detection

## Output

### Video Output
- Unique colored bounding boxes for each cone
- Cone IDs displayed above boxes
- Confidence scores
- Optional movement trails
- Tracking statistics overlay

### JSON Export (--save-tracks)
```json
{
  "frame_count": 1524,
  "tracks": [
    {
      "track_id": 0,
      "history": [
        {
          "frame": 1,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.85
        }
      ],
      "final_confidence": 0.95,
      "total_age": 1500
    }
  ]
}
```

## Performance

- Adds ~10-15ms per frame for tracking
- Handles 10+ cones simultaneously
- Robust to 50%+ detection failure rate
- Maintains stable IDs through camera movement

## Tips

1. **Adjust Detection Confidence**: Lower `-c` value if cones are missed
2. **Preview First**: Use `-p` to verify tracking before processing entire video
3. **Export Data**: Use `--save-tracks` for post-processing analysis
4. **Trail Length**: Trails show last 50 positions (customizable in code)

## Limitations

- Requires consistent cone appearance
- Very fast camera movements may challenge tracking
- Overlapping cones may swap IDs temporarily

## Example Scenarios

### Training Session Analysis
Track cone positions during drills to analyze player movement patterns:
```bash
python simple_detector.py training.mov -m model.onnx --imgsz 960 --track-cones --save-tracks drill_analysis.json
```

### Game Setup Verification
Ensure cones remain in position during game setup:
```bash
python simple_detector.py setup.mov -m model.onnx --imgsz 960 --track-cones --show-trails -p
```