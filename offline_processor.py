import cv2
import numpy as np
import os
import json
import time
from collections import defaultdict, Counter
from datetime import datetime
import argparse
from scipy import interpolate
from scipy.stats import mode

from ultralytics import YOLO
from cone_tracker import ConeTracker
from simple_detector import load_model_with_classes, get_unique_color


class TemporalContext:
    """Stores temporal information about detections across the entire video"""
    
    def __init__(self):
        self.frame_detections = []  # All detections for each frame
        self.object_trajectories = defaultdict(list)  # Object ID -> list of (frame, bbox, class)
        self.scene_statistics = {}
        self.total_frames = 0
        
    def add_frame_detections(self, frame_idx, detections):
        """Add detections for a specific frame"""
        self.frame_detections.append({
            'frame_idx': frame_idx,
            'detections': detections
        })
        self.total_frames = max(self.total_frames, frame_idx + 1)
    
    def get_temporal_window(self, frame_idx, window_size=10):
        """Get detections from surrounding frames"""
        start = max(0, frame_idx - window_size)
        end = min(self.total_frames, frame_idx + window_size + 1)
        
        past = [self.frame_detections[i] for i in range(start, frame_idx) 
                if i < len(self.frame_detections)]
        future = [self.frame_detections[i] for i in range(frame_idx + 1, end) 
                  if i < len(self.frame_detections)]
        
        return past, future
    
    def analyze_scene_statistics(self):
        """Analyze global statistics from all detections"""
        cone_counts = []
        football_counts = []
        cone_positions = []
        
        for frame_data in self.frame_detections:
            cones = [d for d in frame_data['detections'] if d['class_name'] == 'cone']
            footballs = [d for d in frame_data['detections'] if d['class_name'] == 'football']
            
            cone_counts.append(len(cones))
            football_counts.append(len(footballs))
            
            for cone in cones:
                center = ((cone['bbox'][0] + cone['bbox'][2]) / 2,
                         (cone['bbox'][1] + cone['bbox'][3]) / 2)
                cone_positions.append(center)
        
        # Find the most common cone count
        if cone_counts:
            most_common_cone_count = Counter(cone_counts).most_common(1)[0][0]
        else:
            most_common_cone_count = 0
            
        self.scene_statistics = {
            'expected_cone_count': most_common_cone_count,
            'cone_count_distribution': Counter(cone_counts),
            'football_detection_rate': sum(football_counts) / len(football_counts) if football_counts else 0,
            'stable_cone_positions': self._find_stable_positions(cone_positions)
        }
    
    def _find_stable_positions(self, positions, cluster_threshold=50):
        """Find positions where cones appear consistently"""
        if not positions:
            return []
            
        # Simple clustering to find stable cone positions
        clusters = []
        for pos in positions:
            added = False
            for cluster in clusters:
                center = np.mean(cluster['positions'], axis=0)
                if np.linalg.norm(np.array(pos) - center) < cluster_threshold:
                    cluster['positions'].append(pos)
                    cluster['count'] += 1
                    added = True
                    break
            
            if not added:
                clusters.append({
                    'positions': [pos],
                    'count': 1
                })
        
        # Return clusters that appear in many frames
        stable_positions = []
        min_appearances = self.total_frames * 0.3  # Cone should appear in at least 30% of frames
        
        for cluster in clusters:
            if cluster['count'] > min_appearances:
                stable_positions.append({
                    'position': np.mean(cluster['positions'], axis=0),
                    'confidence': cluster['count'] / self.total_frames
                })
        
        return stable_positions


class GlobalVideoAnalyzer:
    """Analyzes entire video in first pass to build temporal context"""
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.temporal_context = TemporalContext()
        
    def analyze_video(self, video_path, conf_threshold=0.5):
        """First pass: analyze entire video to build context"""
        print("Pass 1: Analyzing entire video...")
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            if hasattr(self.model, 'imgsz') and self.model.imgsz:
                results = self.model(frame, conf=conf_threshold, verbose=False, imgsz=self.model.imgsz)
            else:
                results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Process detections
            frame_detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            class_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                            class_name = self.class_names.get(class_id, f'class_{class_id}')
                            
                            # Handle class mapping
                            if class_name == 'object_0' or (class_id == 0 and class_name in ['object_0', 'cone']):
                                class_name = 'cone'
                            elif class_name == 'object_1' or (class_id == 1 and class_name in ['object_1', 'football']):
                                class_name = 'football'
                            
                            frame_detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'class_name': class_name,
                                'class_id': class_id
                            })
            
            self.temporal_context.add_frame_detections(frame_idx, frame_detections)
            
            # Progress update
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"\rPass 1 Progress: {progress:.1f}% | Frame: {frame_idx}/{total_frames} | "
                      f"FPS: {fps:.1f}", end='', flush=True)
            
            frame_idx += 1
        
        cap.release()
        print(f"\nPass 1 complete. Analyzed {frame_idx} frames.")
        
        # Analyze scene statistics
        self.temporal_context.analyze_scene_statistics()
        print(f"Scene statistics: {self.temporal_context.scene_statistics}")
        
        return self.temporal_context


class TemporalValidator:
    """Validates and corrects detections using temporal consistency"""
    
    def __init__(self, temporal_context, validation_window=5):
        self.temporal_context = temporal_context
        self.validation_window = validation_window
        
    def validate_classification(self, frame_idx, detection, past, future):
        """Validate if a detection's class is consistent with temporal context"""
        # Get similar detections in nearby frames
        similar_detections = []
        
        # Check past frames
        for past_frame in past[-self.validation_window:]:
            for past_det in past_frame['detections']:
                if self._is_same_object(detection['bbox'], past_det['bbox']):
                    similar_detections.append(past_det['class_name'])
        
        # Check future frames
        for future_frame in future[:self.validation_window]:
            for future_det in future_frame['detections']:
                if self._is_same_object(detection['bbox'], future_det['bbox']):
                    similar_detections.append(future_det['class_name'])
        
        if not similar_detections:
            return detection['class_name'], detection['confidence']
        
        # Find majority class
        class_counts = Counter(similar_detections)
        majority_class = class_counts.most_common(1)[0][0]
        consistency_score = class_counts[majority_class] / len(similar_detections)
        
        # Override classification if inconsistent
        if detection['class_name'] != majority_class and consistency_score > 0.7:
            return majority_class, detection['confidence'] * consistency_score
        
        return detection['class_name'], detection['confidence']
    
    def _is_same_object(self, bbox1, bbox2, iou_threshold=0.5):
        """Check if two bboxes likely represent the same object"""
        # Calculate IoU
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-6)
        return iou > iou_threshold
    
    def validate_cone_count(self, frame_detections):
        """Check if cone count is consistent with expected"""
        expected_count = self.temporal_context.scene_statistics['expected_cone_count']
        cone_count = sum(1 for d in frame_detections if d['class_name'] == 'cone')
        
        # Flag as anomaly if count differs significantly
        if abs(cone_count - expected_count) > 2:
            return False, f"Unusual cone count: {cone_count} (expected: {expected_count})"
        
        return True, "Normal cone count"


class BidirectionalTracker:
    """Tracks objects both forward and backward through video"""
    
    def __init__(self, temporal_context):
        self.temporal_context = temporal_context
        self.forward_tracks = {}
        self.backward_tracks = {}
        
    def track_bidirectional(self):
        """Perform forward and backward tracking"""
        print("\nPerforming bidirectional tracking...")
        
        # Forward pass
        self.forward_tracks = self._track_forward()
        
        # Backward pass
        self.backward_tracks = self._track_backward()
        
        # Merge tracks
        merged_tracks = self._merge_tracks()
        
        return merged_tracks
    
    def _track_forward(self):
        """Track objects from start to end"""
        tracker = ConeTracker(max_age=30, min_hits=3)
        tracks = defaultdict(list)
        
        for frame_data in self.temporal_context.frame_detections:
            frame_idx = frame_data['frame_idx']
            cone_detections = [d['bbox'] for d in frame_data['detections'] 
                              if d['class_name'] == 'cone']
            
            tracked_cones = tracker.update(cone_detections)
            
            for cone in tracked_cones:
                tracks[cone['track_id']].append({
                    'frame': frame_idx,
                    'bbox': cone['bbox'],
                    'confidence': cone['confidence']
                })
        
        return tracks
    
    def _track_backward(self):
        """Track objects from end to start"""
        tracker = ConeTracker(max_age=30, min_hits=3)
        tracks = defaultdict(list)
        
        # Process frames in reverse order
        for frame_data in reversed(self.temporal_context.frame_detections):
            frame_idx = frame_data['frame_idx']
            cone_detections = [d['bbox'] for d in frame_data['detections'] 
                              if d['class_name'] == 'cone']
            
            tracked_cones = tracker.update(cone_detections)
            
            for cone in tracked_cones:
                tracks[cone['track_id']].append({
                    'frame': frame_idx,
                    'bbox': cone['bbox'],
                    'confidence': cone['confidence']
                })
        
        # Reverse the frame order for each track
        for track_id in tracks:
            tracks[track_id].reverse()
        
        return tracks
    
    def _merge_tracks(self):
        """Merge forward and backward tracks"""
        merged = {}
        
        # Find corresponding tracks between forward and backward passes
        for fwd_id, fwd_track in self.forward_tracks.items():
            best_match = None
            best_overlap = 0
            
            for bwd_id, bwd_track in self.backward_tracks.items():
                overlap = self._calculate_track_overlap(fwd_track, bwd_track)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = bwd_id
            
            if best_match and best_overlap > 0.5:
                # Merge the tracks
                merged[fwd_id] = self._merge_two_tracks(fwd_track, 
                                                        self.backward_tracks[best_match])
            else:
                merged[fwd_id] = fwd_track
        
        return merged
    
    def _calculate_track_overlap(self, track1, track2):
        """Calculate overlap between two tracks"""
        frames1 = {t['frame'] for t in track1}
        frames2 = {t['frame'] for t in track2}
        
        common_frames = frames1.intersection(frames2)
        if not common_frames:
            return 0
        
        # Calculate average IoU over common frames
        total_iou = 0
        for frame in common_frames:
            bbox1 = next(t['bbox'] for t in track1 if t['frame'] == frame)
            bbox2 = next(t['bbox'] for t in track2 if t['frame'] == frame)
            
            # Calculate IoU
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            
            iou = intersection / (union + 1e-6)
            total_iou += iou
        
        return total_iou / len(common_frames)
    
    def _merge_two_tracks(self, track1, track2):
        """Merge two tracks by averaging positions"""
        merged = []
        
        frames1 = {t['frame']: t for t in track1}
        frames2 = {t['frame']: t for t in track2}
        
        all_frames = sorted(set(frames1.keys()).union(frames2.keys()))
        
        for frame in all_frames:
            if frame in frames1 and frame in frames2:
                # Average the bboxes
                bbox1 = frames1[frame]['bbox']
                bbox2 = frames2[frame]['bbox']
                avg_bbox = [(b1 + b2) / 2 for b1, b2 in zip(bbox1, bbox2)]
                
                merged.append({
                    'frame': frame,
                    'bbox': avg_bbox,
                    'confidence': (frames1[frame]['confidence'] + 
                                 frames2[frame]['confidence']) / 2
                })
            elif frame in frames1:
                merged.append(frames1[frame])
            else:
                merged.append(frames2[frame])
        
        return merged


class GapInterpolator:
    """Interpolates missing detections using future and past information"""
    
    def __init__(self, max_gap_size=30):
        self.max_gap_size = max_gap_size
        
    def interpolate_tracks(self, tracks):
        """Fill gaps in tracks using interpolation"""
        interpolated_tracks = {}
        
        for track_id, track_data in tracks.items():
            if len(track_data) < 2:
                interpolated_tracks[track_id] = track_data
                continue
            
            # Find gaps
            frames = [t['frame'] for t in track_data]
            gaps = []
            
            for i in range(1, len(frames)):
                if frames[i] - frames[i-1] > 1:
                    gaps.append((frames[i-1], frames[i]))
            
            # Interpolate gaps
            interpolated = track_data.copy()
            
            for start_frame, end_frame in gaps:
                gap_size = end_frame - start_frame - 1
                
                if gap_size <= self.max_gap_size:
                    # Get positions to interpolate between
                    start_data = next(t for t in track_data if t['frame'] == start_frame)
                    end_data = next(t for t in track_data if t['frame'] == end_frame)
                    
                    # Interpolate bbox positions
                    for frame in range(start_frame + 1, end_frame):
                        t = (frame - start_frame) / (end_frame - start_frame)
                        
                        interpolated_bbox = [
                            start_data['bbox'][i] * (1 - t) + end_data['bbox'][i] * t
                            for i in range(4)
                        ]
                        
                        interpolated.append({
                            'frame': frame,
                            'bbox': interpolated_bbox,
                            'confidence': min(start_data['confidence'], 
                                            end_data['confidence']) * 0.8,
                            'interpolated': True
                        })
            
            # Sort by frame
            interpolated_tracks[track_id] = sorted(interpolated, 
                                                   key=lambda x: x['frame'])
        
        return interpolated_tracks


class OfflineVideoProcessor:
    """Main class for offline video processing with temporal analysis"""
    
    def __init__(self, model_path, conf_threshold=0.5, imgsz=None):
        # Load model
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        if imgsz and model_path.endswith('.onnx'):
            self.model = YOLO(model_path, task='detect')
            self.model.imgsz = imgsz
            self.class_names = {0: 'cone', 1: 'football'}
        else:
            self.model, self.class_names = load_model_with_classes(model_path)
        
        # Initialize components
        self.global_analyzer = GlobalVideoAnalyzer(self.model, self.class_names)
        self.temporal_context = None
        self.validator = None
        self.bidirectional_tracker = None
        self.interpolator = GapInterpolator()
        
    def process_video(self, video_path, output_path=None, two_pass=True, 
                      bidirectional=True, interpolate_gaps=True):
        """Process video with full temporal analysis"""
        
        # Pass 1: Global analysis
        self.temporal_context = self.global_analyzer.analyze_video(video_path, 
                                                                  self.conf_threshold)
        self.validator = TemporalValidator(self.temporal_context)
        
        if bidirectional:
            # Bidirectional tracking
            self.bidirectional_tracker = BidirectionalTracker(self.temporal_context)
            tracks = self.bidirectional_tracker.track_bidirectional()
            
            if interpolate_gaps:
                # Interpolate missing detections
                tracks = self.interpolator.interpolate_tracks(tracks)
        
        # Pass 2: Process with temporal context
        if two_pass:
            print("\nPass 2: Processing with temporal context...")
            self._process_with_context(video_path, output_path, tracks)
        
        return tracks
    
    def _process_with_context(self, video_path, output_path, tracks):
        """Second pass processing with temporal validation"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_offline_processed.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        start_time = time.time()
        
        # Convert tracks to frame-based lookup
        frame_tracks = defaultdict(list)
        for track_id, track_data in tracks.items():
            for data in track_data:
                frame_tracks[data['frame']].append({
                    'track_id': track_id,
                    'bbox': data['bbox'],
                    'confidence': data['confidence'],
                    'interpolated': data.get('interpolated', False)
                })
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get temporal context
            past, future = self.temporal_context.get_temporal_window(frame_idx)
            
            # Get tracks for this frame
            current_tracks = frame_tracks.get(frame_idx, [])
            
            # Draw tracked cones
            for track_data in current_tracks:
                track_id = track_data['track_id']
                bbox = track_data['bbox']
                confidence = track_data['confidence']
                is_interpolated = track_data['interpolated']
                
                x1, y1, x2, y2 = [int(b) for b in bbox]
                color = get_unique_color(track_id)
                
                # Draw differently for interpolated positions
                if is_interpolated:
                    # Dashed line for interpolated
                    thickness = 1
                    dash_length = 5
                    for i in range(0, x2-x1, dash_length*2):
                        cv2.line(frame, (x1+i, y1), (min(x1+i+dash_length, x2), y1), color, thickness)
                        cv2.line(frame, (x1+i, y2), (min(x1+i+dash_length, x2), y2), color, thickness)
                else:
                    # Solid line for detected
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"Cone {track_id + 1}: {confidence:.2f}"
                if is_interpolated:
                    label += " (interp)"
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1),
                            color, -1)
                cv2.putText(frame, label,
                           (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1)
            
            # Draw footballs from original detections
            if frame_idx < len(self.temporal_context.frame_detections):
                frame_detections = self.temporal_context.frame_detections[frame_idx]['detections']
                for det in frame_detections:
                    if det['class_name'] == 'football':
                        x1, y1, x2, y2 = [int(b) for b in det['bbox']]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        label = f"Football: {det['confidence']:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, 
                                    (x1, y1 - label_size[1] - 4),
                                    (x1 + label_size[0], y1),
                                    (0, 255, 0), -1)
                        cv2.putText(frame, label,
                                   (x1, y1 - 2),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.6, (255, 255, 255), 2)
            
            # Add info overlay
            info_text = f"Frame: {frame_idx}/{total_frames} | Cones: {len(current_tracks)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Progress update
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                progress = (frame_idx / total_frames) * 100
                print(f"\rPass 2 Progress: {progress:.1f}% | Frame: {frame_idx}/{total_frames} | "
                      f"FPS: {fps_proc:.1f}", end='', flush=True)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        print(f"\n\nProcessing complete! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Offline video processing with temporal analysis')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('-m', '--model', help='Path to YOLO model file', required=True)
    parser.add_argument('-o', '--output', help='Path to output video file', default=None)
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Input image size for ONNX models')
    
    # Processing options
    parser.add_argument('--two-pass', action='store_true', default=True,
                       help='Use two-pass processing (default: True)')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='Use bidirectional tracking (default: True)')
    parser.add_argument('--interpolate-gaps', action='store_true', default=True,
                       help='Interpolate missing detections (default: True)')
    parser.add_argument('--save-analysis', type=str, default=None,
                       help='Save temporal analysis to JSON file')
    
    args = parser.parse_args()
    
    # Create processor
    processor = OfflineVideoProcessor(
        model_path=args.model,
        conf_threshold=args.confidence,
        imgsz=args.imgsz
    )
    
    # Process video
    tracks = processor.process_video(
        video_path=args.video,
        output_path=args.output,
        two_pass=args.two_pass,
        bidirectional=args.bidirectional,
        interpolate_gaps=args.interpolate_gaps
    )
    
    # Save analysis if requested
    if args.save_analysis:
        analysis_data = {
            'scene_statistics': processor.temporal_context.scene_statistics,
            'total_frames': processor.temporal_context.total_frames,
            'tracks': {
                str(track_id): [
                    {
                        'frame': t['frame'],
                        'bbox': t['bbox'],
                        'confidence': t['confidence'],
                        'interpolated': t.get('interpolated', False)
                    }
                    for t in track_data
                ]
                for track_id, track_data in tracks.items()
            }
        }
        
        with open(args.save_analysis, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"Analysis saved to: {args.save_analysis}")


if __name__ == "__main__":
    main()