import cv2
import os
import sys
import platform
import argparse
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from cone_tracker import ConeTracker


def load_model_with_classes(model_path):
    """Load model and handle class mapping properly"""
    try:
        # Check if model file is valid
        if os.path.getsize(model_path) < 1024:
            raise ValueError(f"Model file is too small. Please use a real model file.")
        
        print(f"Loading model: {os.path.basename(model_path)}")
        
        # For ONNX models, we need to check input dimensions
        if model_path.endswith('.onnx'):
            # Try to load with ONNX Runtime first to get model info
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(model_path)
                input_shape = session.get_inputs()[0].shape
                # Shape is usually [batch_size, channels, height, width]
                if len(input_shape) >= 4:
                    imgsz = input_shape[2]  # Assuming square input
                    print(f"ONNX model expects input size: {imgsz}x{imgsz}")
                else:
                    imgsz = 640  # Default
                    
                # Load model first, then we'll set size during inference
                model = YOLO(model_path, task='detect')
                model.imgsz = imgsz  # Store for later use
                print(f"Loaded ONNX model, will use imgsz={imgsz} during inference")
            except:
                # Fallback to default
                model = YOLO(model_path, task='detect')
                model.imgsz = 960  # Store for later use
                print("Loaded ONNX model, will use imgsz=960 during inference (fallback)")
        else:
            model = YOLO(model_path)
            print("Loaded PyTorch model")
        
        # Try to detect model type and set class names
        filename = os.path.basename(model_path).lower()
        
        # Check if model has names attribute
        if hasattr(model, 'names') and model.names:
            print(f"Model has built-in classes: {model.names}")
            return model, model.names
        
        # For ONNX models or models without names, set defaults based on filename
        if 'football' in filename or 'cone' in filename:
            class_names = {0: 'cone', 1: 'football'}  # Based on your model
            print(f"Using cone/football class mapping")
        elif 'trtfootballyolo' in filename:
            class_names = {0: 'cone', 1: 'football'}
            print(f"Using TRT football model class mapping")
        else:
            # Try to infer by running a test inference
            try:
                # Create dummy image for test
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                results = model(dummy_image, verbose=False)
                
                # If we can get any info from results, use it
                if results and hasattr(results[0], 'names'):
                    class_names = results[0].names
                    print(f"Detected classes from test inference: {class_names}")
                else:
                    # Default fallback - assume object_0 is cone, object_1 is football
                    class_names = {0: 'cone', 1: 'football'}
                    print(f"Using default mapping: 0=cone, 1=football")
            except:
                class_names = {0: 'cone', 1: 'football'}
                print(f"Using default mapping: 0=cone, 1=football (fallback)")
        
        return model, class_names
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def get_unique_color(track_id):
    """Generate a unique color for each track ID"""
    # Use golden ratio to generate visually distinct colors
    golden_ratio_conjugate = 0.618033988749895
    hue = (track_id * golden_ratio_conjugate) % 1.0
    
    # Convert HSV to RGB
    import colorsys
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return tuple(int(c * 255) for c in rgb[::-1])  # BGR for OpenCV


def process_video_simple(video_path, model_path=None, output_path=None, conf_threshold=0.5, show_preview=False, imgsz=None, track_cones=False, show_trails=False, save_tracks=None, strict_validation=True, football_buffer=50):
    """
    Process video with simple object detection (no tracking).
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model (optional, will use yolov8n.pt by default)
        output_path: Path to output video (optional)
        conf_threshold: Confidence threshold for detections
        show_preview: Whether to show live preview
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Load model - require model path
    if not model_path:
        print("Error: Model path is required. Use -m to specify your model file.")
        return False
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    # Load model with proper class handling
    try:
        # If user specified image size, use it
        if imgsz and model_path.endswith('.onnx'):
            print(f"Using user-specified image size: {imgsz}")
            model = YOLO(model_path, task='detect')
            model.imgsz = imgsz  # Store for use during inference
            # Still need to get class names - assume 0=cone, 1=football based on your model
            filename = os.path.basename(model_path).lower()
            class_names = {0: 'cone', 1: 'football'}
            print(f"Using cone/football mapping for user-specified size")
        else:
            model, class_names = load_model_with_classes(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTip: If you're getting dimension errors, try specifying --imgsz 960")
        return False
    
    # Create output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path) or '.'
        output_path = os.path.join(output_dir, f"{base_name}_detected.mp4")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    
    # macOS preview warning
    if show_preview and platform.system() == 'Darwin':
        print("\nNote: Preview on macOS may have performance limitations.")
        print("Press 'q' or ESC to stop processing early.")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize cone tracker if requested
    cone_tracker = None
    cone_trails = {}  # Store trails for each cone
    if track_cones:
        cone_tracker = ConeTracker(
            max_age=30, 
            min_hits=3, 
            iou_threshold=0.3,
            min_distance_from_football=football_buffer if strict_validation else 0,
            cone_velocity_threshold=5
        )
        print(f"Cone tracking enabled (strict validation: {strict_validation}, football buffer: {football_buffer}px)")
    
    # Process video
    print("\nProcessing video...")
    frame_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection with timing
            start_inference = time.time()
            # Use the image size if set on the model
            if hasattr(model, 'imgsz') and model.imgsz:
                results = model(frame, conf=conf_threshold, verbose=False, imgsz=model.imgsz)
            else:
                results = model(frame, conf=conf_threshold, verbose=False)
            inference_time = (time.time() - start_inference) * 1000
            
            # Process results
            detection_count = 0
            cone_detections = []
            football_detections = []
            other_detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                # First, collect all detections
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            class_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                            class_name = class_names.get(class_id, f'class_{class_id}')
                            
                            detection_info = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_name': class_name
                            }
                            
                            # Separate cones from other objects
                            # Check if it's a cone by class name or class ID
                            is_cone = (class_name.lower() in ['cone', 'marker', 'object_0'] or 
                                      (class_id == 0 and class_name == 'object_0'))
                            is_football = (class_name.lower() in ['football', 'ball', 'soccer_ball', 'object_1'] or
                                          (class_id == 1 and class_name == 'object_1'))
                            
                            if is_football:
                                detection_info['class_name'] = 'football'
                                football_detections.append(detection_info)
                            elif is_cone and track_cones:
                                cone_detections.append(detection_info['bbox'])
                            else:
                                # Update class name for display
                                if is_cone and not track_cones:
                                    detection_info['class_name'] = 'cone'
                                other_detections.append(detection_info)
                            
                            detection_count += 1
            
            # Update football position in tracker BEFORE processing cones
            if track_cones and cone_tracker and football_detections:
                # Use the highest confidence football detection
                best_football = max(football_detections, key=lambda x: x['confidence'])
                cone_tracker.update_football(best_football['bbox'])
            
            # Track cones if enabled
            tracked_cones = []
            if track_cones and cone_tracker:
                tracked_cones = cone_tracker.update(cone_detections)
                
                # Update trails
                for cone in tracked_cones:
                    track_id = cone['track_id']
                    bbox = cone['bbox']
                    # Ensure bbox values are integers
                    center = (int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2))
                    
                    if track_id not in cone_trails:
                        cone_trails[track_id] = []
                    cone_trails[track_id].append(center)
                    
                    # Limit trail length
                    if len(cone_trails[track_id]) > 50:
                        cone_trails[track_id].pop(0)
            
            # Draw trails if requested
            if show_trails and track_cones:
                for track_id, trail in cone_trails.items():
                    if len(trail) > 1:
                        color = get_unique_color(track_id)
                        for i in range(1, len(trail)):
                            # Convert to tuples for cv2.line
                            pt1 = tuple(trail[i-1]) if isinstance(trail[i-1], (list, np.ndarray)) else trail[i-1]
                            pt2 = tuple(trail[i]) if isinstance(trail[i], (list, np.ndarray)) else trail[i]
                            cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw tracked cones
            for cone in tracked_cones:
                bbox = cone['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                track_id = cone['track_id']
                confidence = cone['confidence']
                
                # Get unique color for this cone
                color = get_unique_color(track_id)
                
                # Draw bounding box (solid for confirmed, dashed for tentative)
                if cone['confirmed']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    # Draw dashed rectangle for tentative tracks
                    dash_length = 10
                    for i in range(0, x2-x1, dash_length*2):
                        cv2.line(frame, (x1+i, y1), (min(x1+i+dash_length, x2), y1), color, 2)
                        cv2.line(frame, (x1+i, y2), (min(x1+i+dash_length, x2), y2), color, 2)
                    for i in range(0, y2-y1, dash_length*2):
                        cv2.line(frame, (x1, y1+i), (x1, min(y1+i+dash_length, y2)), color, 2)
                        cv2.line(frame, (x2, y1+i), (x2, min(y1+i+dash_length, y2)), color, 2)
                
                # Draw label with prominent ID (1-indexed for user friendliness)
                label = f"Cone {track_id + 1}"
                confidence_label = f"{confidence:.2f}"
                
                # Larger font for ID
                font_scale = 0.8
                font_thickness = 2
                
                # Get text sizes
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                conf_size, _ = cv2.getTextSize(confidence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw ID label background
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 8),
                            (x1 + label_size[0] + 4, y1),
                            color, -1)
                
                # Draw ID text in white
                cv2.putText(frame, label,
                           (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale, (255, 255, 255), font_thickness)
                
                # Draw confidence below ID (smaller)
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - conf_size[1] - 12),
                            (x1 + conf_size[0] + 4, y1 - label_size[1] - 8),
                            (0, 0, 0), -1)
                
                cv2.putText(frame, confidence_label,
                           (x1 + 2, y1 - label_size[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1)
                
                # Also draw ID in center of cone for better visibility
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                id_text = f"{track_id + 1}"  # 1-indexed
                id_size, _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                
                # Draw white circle background for center ID
                cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), -1)
                cv2.circle(frame, (center_x, center_y), 22, color, 2)
                
                # Draw ID number in center
                cv2.putText(frame, id_text,
                           (center_x - id_size[0]//2, center_y + id_size[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.0, color, 2)
            
            # Draw football detections with special visualization
            for det in football_detections:
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                
                # Bright green for football
                color = (0, 255, 0)
                
                # Thicker box for football
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw football label
                label = f"Football: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                cv2.putText(frame, label,
                           (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 2)
                
                # Add motion indicator if tracking
                if track_cones and cone_tracker and len(cone_tracker.football_history) > 1:
                    # Draw small trail for football
                    for i in range(max(0, len(cone_tracker.football_history) - 5), len(cone_tracker.football_history)):
                        _, fb = cone_tracker.football_history[i]
                        center = ((fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2)
                        cv2.circle(frame, center, 3, (0, 200, 0), -1)
            
            # Draw other (non-cone, non-football) detections
            for det in other_detections:
                x1, y1, x2, y2 = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                
                color = (255, 0, 0)  # Blue for others
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                cv2.putText(frame, label,
                           (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1)
            
            # Add tracking summary if enabled
            if track_cones and cone_tracker:
                summary = cone_tracker.get_tracks_summary()
                text = f"Tracked Cones: {summary['confirmed_tracks']} confirmed, {summary['tentative_tracks']} tentative"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Show preview if requested (every other frame for efficiency)
            if show_preview and frame_count % 2 == 0:
                # Resize frame for preview to ensure it fits on screen
                preview_height = 720
                scale = preview_height / height
                preview_width = int(width * scale)
                preview_frame = cv2.resize(frame, (preview_width, preview_height))
                
                cv2.imshow('Object Detection', preview_frame)
                # Increase wait time for macOS compatibility
                key = cv2.waitKey(1) & 0xFF  # Reduced wait time since showing every other frame
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nProcessing interrupted by user")
                    break
            
            # Update progress
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
                
                if track_cones and cone_tracker:
                    summary = cone_tracker.get_tracks_summary()
                    print(f"\rProgress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                          f"FPS: {fps_processing:.1f} | Inference: {inference_time:.1f}ms | "
                          f"Tracked Cones: {summary['confirmed_tracks']} | ETA: {eta:.1f}s", end='', flush=True)
                else:
                    print(f"\rProgress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                          f"FPS: {fps_processing:.1f} | Inference: {inference_time:.1f}ms | "
                          f"Detections: {detection_count} | ETA: {eta:.1f}s", end='', flush=True)
        
        print(f"\n\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        return False
        
    finally:
        # Save tracking data if requested
        if save_tracks and cone_tracker:
            cone_tracker.save_tracks(save_tracks)
            print(f"\nTracking data saved to: {save_tracks}")
        
        # Clean up
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Object detection with optional cone tracking')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('-m', '--model', help='Path to YOLO model file (.pt or .onnx)', required=True)
    parser.add_argument('-o', '--output', help='Path to output video file', default=None)
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('-p', '--preview', action='store_true',
                       help='Show live preview during processing (resized to 720p for macOS compatibility)')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Input image size for ONNX models (e.g., 640, 960, 1280)')
    
    # Tracking options
    parser.add_argument('--track-cones', action='store_true',
                       help='Enable cone tracking with unique IDs')
    parser.add_argument('--show-trails', action='store_true',
                       help='Show movement trails for tracked cones')
    parser.add_argument('--save-tracks', type=str, default=None,
                       help='Save tracking data to JSON file')
    parser.add_argument('--strict-validation', action='store_true',
                       help='Enable strict cone validation to prevent football misclassification')
    parser.add_argument('--football-buffer', type=int, default=50,
                       help='Minimum distance from football for new cone tracks (default: 50 pixels)')
    
    args = parser.parse_args()
    
    # Process video
    success = process_video_simple(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        conf_threshold=args.confidence,
        show_preview=args.preview,
        imgsz=args.imgsz,
        track_cones=args.track_cones,
        show_trails=args.show_trails,
        save_tracks=args.save_tracks,
        strict_validation=args.strict_validation,
        football_buffer=args.football_buffer
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()