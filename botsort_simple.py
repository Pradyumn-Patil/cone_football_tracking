#!/usr/bin/env python3
"""
Simple BoT-SORT tracking with YOLOv8 and Ultralytics
Clean implementation without modifications
"""

import cv2
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import json


def process_video(video_path, model_path, output_path=None, show_preview=False, 
                  conf_threshold=0.25, imgsz=960):
    """
    Process video with YOLOv8 and BoT-SORT tracking
    
    Args:
        video_path: Path to input video
        model_path: Path to ONNX model
        output_path: Path to output video (optional)
        show_preview: Whether to show live preview
        conf_threshold: Detection confidence threshold
        imgsz: Input size for ONNX model
    """
    
    # Check inputs
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    print(f"Loading YOUR ONNX model: {model_path}")
    # Load your custom ONNX model (NOT a pretrained YOLO model)
    model = YOLO(model_path, task='detect')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    
    # Create output path if needed
    if output_path is None:
        base_name = Path(video_path).stem
        output_path = f"{base_name}_botsort.mp4"
        
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Track history for trails
    track_history = {}
    
    print("\nProcessing with BoT-SORT...")
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run tracking with YOUR ONNX model + BoT-SORT
            # persist=True maintains track IDs between frames
            results = model.track(
                frame, 
                persist=True,
                conf=conf_threshold,
                iou=0.7,
                imgsz=imgsz,
                tracker="botsort.yaml"  # Use BoT-SORT tracker
            )
            
            # Process results
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    # Get track IDs
                    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        track_ids = result.boxes.id.int().cpu().numpy()
                    else:
                        track_ids = [-1] * len(boxes)
                    
                    # Get classes and confidences
                    class_ids = result.boxes.cls.int().cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    # Draw each detection
                    for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        
                        if track_id >= 0:
                            # Update track history
                            if track_id not in track_history:
                                track_history[track_id] = []
                            
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            track_history[track_id].append(center)
                            
                            # Keep only last 30 points
                            if len(track_history[track_id]) > 30:
                                track_history[track_id].pop(0)
                            
                            # Get color for this track
                            color = get_color(track_id)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label = f"ID:{track_id} ({conf:.2f})"
                            if class_id == 0:  # Cone
                                label = f"Cone {track_id}"
                            elif class_id == 1:  # Football
                                label = f"Football {track_id}"
                                
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Draw trail
                            if len(track_history[track_id]) > 1:
                                for i in range(1, len(track_history[track_id])):
                                    cv2.line(frame, 
                                            track_history[track_id][i-1],
                                            track_history[track_id][i],
                                            color, 2)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count} | BoT-SORT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Show preview if requested
            if show_preview:
                cv2.imshow('BoT-SORT Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            # Progress update
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_processing = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"\rProgress: {progress:.1f}% | FPS: {fps_processing:.1f}", end='')
                
    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
            
    print(f"\n\nDone! Output saved to: {output_path}")
    return True


def get_color(track_id):
    """Generate a unique color for each track ID"""
    np.random.seed(track_id)
    color = np.random.randint(0, 255, size=3)
    return tuple(map(int, color))


def main():
    parser = argparse.ArgumentParser(description='Simple BoT-SORT tracking with YOLOv8')
    parser.add_argument('video', help='Path to input video')
    parser.add_argument('-m', '--model', help='Path to ONNX model', required=True)
    parser.add_argument('-o', '--output', help='Output video path')
    parser.add_argument('-p', '--preview', action='store_true', help='Show preview')
    parser.add_argument('-c', '--confidence', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--imgsz', type=int, default=960,
                       help='Model input size (default: 960)')
    
    args = parser.parse_args()
    
    success = process_video(
        args.video,
        args.model,
        args.output,
        args.preview,
        args.confidence,
        args.imgsz
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()