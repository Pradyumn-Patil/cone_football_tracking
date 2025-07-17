import numpy as np
import cv2
from datetime import datetime
import os


def calculate_speed(point1, point2, fps=30):
    """
    Calculate speed between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        fps: Frames per second of the video
    
    Returns:
        Speed in pixels per second
    """
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    speed = distance * fps  # Convert from pixels/frame to pixels/second
    return speed


def smooth_trajectory(points, window_size=5):
    """
    Smooth a trajectory using moving average.
    
    Args:
        points: List of (x, y) points
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed points
    """
    if len(points) < window_size:
        return points
    
    points_array = np.array(points)
    smoothed = np.zeros_like(points_array)
    
    # Apply moving average
    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        smoothed[i] = np.mean(points_array[start:end], axis=0)
    
    return smoothed.tolist()


def create_output_path(input_path, suffix='_tracked'):
    """
    Create output file path based on input path.
    
    Args:
        input_path: Path to input video
        suffix: Suffix to add to the filename
    
    Returns:
        Output file path
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_name = f"{base_name}{suffix}.mp4"
    output_dir = os.path.dirname(input_path)
    return os.path.join(output_dir, output_name)


def get_video_properties(video_path):
    """
    Get video properties.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    properties = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    cap.release()
    return properties


def print_progress(current_frame, total_frames, start_time=None):
    """
    Print processing progress.
    
    Args:
        current_frame: Current frame number
        total_frames: Total number of frames
        start_time: Start time of processing
    """
    progress = (current_frame / total_frames) * 100
    
    if start_time:
        elapsed = (datetime.now() - start_time).total_seconds()
        fps = current_frame / elapsed if elapsed > 0 else 0
        eta = (total_frames - current_frame) / fps if fps > 0 else 0
        
        print(f"\rProgress: {progress:.1f}% | Frame: {current_frame}/{total_frames} | "
              f"FPS: {fps:.1f} | ETA: {eta:.1f}s", end='', flush=True)
    else:
        print(f"\rProgress: {progress:.1f}% | Frame: {current_frame}/{total_frames}", 
              end='', flush=True)


def validate_model_path(model_path):
    """
    Validate that the model file exists.
    
    Args:
        model_path: Path to model file
    
    Returns:
        True if valid, raises exception otherwise
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.onnx'):
        raise ValueError(f"Model file must be ONNX format (.onnx): {model_path}")
    
    return True


def validate_video_path(video_path):
    """
    Validate that the video file exists and is readable.
    
    Args:
        video_path: Path to video file
    
    Returns:
        True if valid, raises exception otherwise
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Try to open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    cap.release()
    return True