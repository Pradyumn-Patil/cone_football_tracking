import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import json


class ConeTrack:
    """Represents a single cone being tracked with Kalman filter"""
    
    def __init__(self, track_id, initial_bbox):
        self.track_id = track_id
        self.bbox = initial_bbox
        
        # Track state
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confidence = 0.3  # Start with low confidence
        self.history = []
        
        # Initialize Kalman filter for center point tracking
        self.kf = self._init_kalman_filter(initial_bbox)
        
        # Store initial position
        self.update_bbox(initial_bbox)
        
    def _init_kalman_filter(self, bbox):
        """Initialize Kalman filter for tracking cone center and size"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        # cx, cy: center position
        # w, h: width and height
        # vx, vy, vw, vh: velocities
        
        # Initial state from bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # cx = cx + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # cy = cy + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ])
        
        # Measurement matrix (we observe cx, cy, w, h)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(4) * 10  # Measurement uncertainty
        
        # Process noise
        q = Q_discrete_white_noise(dim=2, dt=1, var=0.1)
        kf.Q = np.zeros((8, 8))
        kf.Q[0:2, 0:2] = q
        kf.Q[2:4, 2:4] = q * 0.01  # Less noise for size
        kf.Q[4:6, 4:6] = q
        kf.Q[6:8, 6:8] = q * 0.01
        
        # Initial covariance
        kf.P *= 100
        
        return kf
    
    def predict(self):
        """Predict next position using Kalman filter"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        # Update confidence based on track age and recent updates
        if self.time_since_update > 1:
            self.confidence *= 0.95  # Decay confidence when not updated
        
        # Get predicted bbox from Kalman state
        self.bbox = self._state_to_bbox(self.kf.x)
        
    def update_bbox(self, bbox):
        """Update track with new detection"""
        # Prepare measurement
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h])
        
        # Update Kalman filter
        self.kf.update(z)
        
        # Update track state
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        
        # Update confidence
        self.confidence = min(0.95, self.confidence + 0.1)
        
        # Add to history
        self.history.append({
            'frame': self.age,
            'bbox': bbox.tolist(),
            'confidence': self.confidence
        })
        
        # Limit history size
        if len(self.history) > 100:
            self.history.pop(0)
    
    def _state_to_bbox(self, state):
        """Convert Kalman filter state to bbox format"""
        cx, cy, w, h = state[0:4]
        return np.array([
            cx - w/2,
            cy - h/2,
            cx + w/2,
            cy + h/2
        ])
    
    def get_state(self):
        """Get current state for visualization"""
        return {
            'id': self.track_id,
            'bbox': self.bbox.tolist(),
            'confidence': self.confidence,
            'age': self.age,
            'time_since_update': self.time_since_update
        }


class ConeTracker:
    """Main tracker for managing multiple cone tracks with football awareness"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3,
                 min_distance_from_football=50, cone_velocity_threshold=5):
        self.max_age = max_age  # Remove tracks not seen for this many frames
        self.min_hits = min_hits  # Minimum hits before track is confirmed
        self.iou_threshold = iou_threshold  # Minimum IoU for matching
        self.min_distance_from_football = min_distance_from_football
        self.cone_velocity_threshold = cone_velocity_threshold
        
        self.tracks = []
        self.track_id_count = 0
        self.frame_count = 0
        
        # Football tracking
        self.football_history = []  # List of (frame, bbox) tuples
        self.football_history_length = 30  # Keep last 30 frames
        
        # Cone size statistics for validation
        self.cone_sizes = []  # Track typical cone sizes
        self.max_cone_size_variance = 0.3  # 30% size variance allowed
        
    def update_football(self, football_bbox):
        """Update football position history"""
        if football_bbox is not None:
            self.football_history.append((self.frame_count, football_bbox))
            # Keep only recent history
            if len(self.football_history) > self.football_history_length:
                self.football_history.pop(0)
    
    def _is_near_recent_football(self, bbox, frames_back=10):
        """Check if bbox is near any recent football position"""
        if not self.football_history:
            return False
        
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
        # Check recent football positions
        recent_frames = self.frame_count - frames_back
        for frame, football_bbox in self.football_history:
            if frame >= recent_frames:
                football_center = np.array([
                    (football_bbox[0] + football_bbox[2]) / 2,
                    (football_bbox[1] + football_bbox[3]) / 2
                ])
                distance = np.linalg.norm(bbox_center - football_center)
                if distance < self.min_distance_from_football:
                    return True
        
        return False
    
    def _validate_cone_size(self, bbox):
        """Check if bbox size is consistent with typical cone size"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        size = width * height
        
        # If we have established cone sizes
        if len(self.cone_sizes) >= 3:
            avg_size = np.mean(self.cone_sizes)
            if abs(size - avg_size) / avg_size > self.max_cone_size_variance:
                return False
        
        return True
    
    def _validate_new_cone(self, bbox):
        """Validate if a detection should create a new cone track"""
        # Check 1: Not near recent football positions
        if self._is_near_recent_football(bbox):
            return False
        
        # Check 2: Size is reasonable for a cone
        if not self._validate_cone_size(bbox):
            return False
        
        return True
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            List of tracked cones with IDs
        """
        self.frame_count += 1
        
        # Convert detections to numpy array
        if len(detections) == 0:
            detections = np.empty((0, 4))
        else:
            detections = np.array(detections)
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for m in matched:
            self.tracks[m[1]].update_bbox(detections[m[0]])
        
        # Create new tracks for unmatched detections (with validation)
        for i in unmatched_dets:
            # Validate before creating new track
            if self._validate_new_cone(detections[i]):
                track = ConeTrack(self.track_id_count, detections[i])
                self.track_id_count += 1
                self.tracks.append(track)
                
                # Update cone size statistics for confirmed tracks
                if track.hits >= self.min_hits:
                    width = detections[i][2] - detections[i][0]
                    height = detections[i][3] - detections[i][1]
                    self.cone_sizes.append(width * height)
                    # Keep only recent sizes
                    if len(self.cone_sizes) > 20:
                        self.cone_sizes.pop(0)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Return confirmed tracks
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.confidence > 0.5:
                results.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'age': track.age,
                    'confirmed': track.hits >= self.min_hits
                })
        
        return results
    
    def _match_detections(self, detections):
        """Match detections to existing tracks using IoU"""
        if len(self.tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), []
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], np.arange(len(self.tracks))
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(
            detections,
            np.array([t.bbox for t in self.tracks])
        )
        
        # Use Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            # Cost matrix (1 - IoU)
            cost_matrix = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_indices = np.column_stack((row_ind, col_ind))
            
            # Filter out matches with IoU below threshold
            matches = []
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                    matches.append(m)
            matches = np.array(matches)
            
            if len(matches) == 0:
                matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.empty((0, 2), dtype=int)
        
        # Find unmatched detections and tracks
        unmatched_detections = []
        for d in range(len(detections)):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_tracks = []
        for t in range(len(self.tracks)):
            if len(matches) == 0 or t not in matches[:, 1]:
                unmatched_tracks.append(t)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_iou_matrix(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
        # Ensure 2D arrays
        if boxes1.ndim == 1:
            boxes1 = boxes1.reshape(1, -1)
        if boxes2.ndim == 1:
            boxes2 = boxes2.reshape(1, -1)
        
        # Expand dimensions for broadcasting
        boxes1 = boxes1[:, np.newaxis, :]  # (N, 1, 4)
        boxes2 = boxes2[np.newaxis, :, :]  # (1, M, 4)
        
        # Calculate intersection
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Calculate union
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Return as 2D matrix - only squeeze the first dimension if needed
        if iou.ndim == 3:
            return iou.squeeze(axis=0)
        return iou
    
    def get_tracks_summary(self):
        """Get summary of all active tracks"""
        return {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': sum(1 for t in self.tracks if t.hits >= self.min_hits),
            'tentative_tracks': sum(1 for t in self.tracks if t.hits < self.min_hits),
            'frame_count': self.frame_count
        }
    
    def save_tracks(self, filename):
        """Save tracking history to JSON file"""
        data = {
            'frame_count': self.frame_count,
            'tracks': []
        }
        
        for track in self.tracks:
            data['tracks'].append({
                'track_id': track.track_id,
                'history': track.history,
                'final_confidence': track.confidence,
                'total_age': track.age
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)