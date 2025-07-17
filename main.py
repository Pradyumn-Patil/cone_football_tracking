import cv2
import argparse
import os
import sys
from datetime import datetime

from detector import YOLODetector
from tracker import ByteTracker
from visualizer import Visualizer
from utils import (
    create_output_path, 
    get_video_properties, 
    print_progress,
    validate_model_path,
    validate_video_path
)


def process_video(video_path, model_path, output_path=None, conf_threshold=0.5, show_preview=False):
    """
    Process video to track football and cones with speed visualization.
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO ONNX model
        output_path: Path to output video (optional)
        conf_threshold: Confidence threshold for detections
        show_preview: Whether to show live preview
    """
    # Validate inputs
    validate_video_path(video_path)
    validate_model_path(model_path)
    
    # Create output path if not provided
    if output_path is None:
        output_path = create_output_path(video_path)
    
    # Get video properties
    video_props = get_video_properties(video_path)
    print(f"Video properties: {video_props['width']}x{video_props['height']} @ {video_props['fps']} FPS")
    print(f"Total frames: {video_props['total_frames']}")
    
    # Initialize components
    print("Initializing detector...")
    detector = YOLODetector(model_path, conf_threshold)
    
    print("Initializing tracker...")
    tracker = ByteTracker(
        track_thresh=0.5,
        match_thresh=0.8,
        max_time_lost=30,
        min_hits=10
    )
    
    print("Initializing visualizer...")
    visualizer = Visualizer(speed_thresholds=(5, 15, 25))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        video_props['fps'],
        (video_props['width'], video_props['height'])
    )
    
    # Process video
    print("\nProcessing video...")
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = detector.detect(frame)
            
            # Track objects
            tracks = tracker.update(detections)
            
            # Visualize results
            annotated_frame = visualizer.draw_tracks(frame.copy(), tracks)
            annotated_frame = visualizer.add_speed_legend(annotated_frame)
            
            # Write frame
            out.write(annotated_frame)
            
            # Show preview if requested
            if show_preview:
                cv2.imshow('Football Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break
            
            # Update progress
            frame_count += 1
            print_progress(frame_count, video_props['total_frames'], start_time)
        
        print("\n\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    
    finally:
        # Clean up
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Track football and cones with speed visualization')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('model', help='Path to YOLO ONNX model file')
    parser.add_argument('-o', '--output', help='Path to output video file', default=None)
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('-p', '--preview', action='store_true',
                       help='Show live preview during processing')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Process video
    try:
        process_video(
            video_path=args.video,
            model_path=args.model,
            output_path=args.output,
            conf_threshold=args.confidence,
            show_preview=args.preview
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()