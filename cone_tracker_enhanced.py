import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import json
import cv2
from collections import deque


class ConeTrack:
    """Enhanced cone track with appearance features and better state management"""
    
    def __init__(self, track_id, initial_bbox, initial_confidence=0.5):
        self.track_id = track_id
        self.bbox = initial_bbox
        self.confidence = initial_confidence
        
        # Track state
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history = deque(maxlen=100)
        
        # Appearance features
        self.appearance_features = deque(maxlen=5)  # Keep last 5 appearance embeddings
        self.color_histogram = None
        
        # Initialize Kalman filter
        self.kf = self._init_kalman_filter(initial_bbox)
        
        # Track states for ByteTrack-style management
        self.state = 'tentative'  # 'tentative', 'confirmed', 'lost'
        self.track_activation_threshold = 0.7
        
        # Track activity status - tracks are never deleted, just marked inactive
        self.is_active = True
        self.last_active_frame = 0
        
        # Store initial position
        self.update_bbox(initial_bbox, initial_confidence, 0)
        
    def _init_kalman_filter(self, bbox):
        """Initialize Kalman filter for tracking cone center and size"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(4) * 10
        
        # Process noise
        q = Q_discrete_white_noise(dim=2, dt=1, var=0.1)
        kf.Q = np.zeros((8, 8))
        kf.Q[0:2, 0:2] = q
        kf.Q[2:4, 2:4] = q * 0.01
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
        
        # Update state based on time since update
        if self.state == 'confirmed' and self.time_since_update > 1:
            self.state = 'lost'
        
        # Decay confidence when not updated
        if self.time_since_update > 1:
            self.confidence *= 0.95
        
        # Get predicted bbox from Kalman state
        self.bbox = self._state_to_bbox(self.kf.x)
        
    def update_bbox(self, bbox, confidence=1.0, frame_count=0):
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
        self.confidence = min(0.95, self.confidence + 0.1)
        
        # Reactivate track if it was inactive
        if not self.is_active:
            print(f"Reactivating track {self.track_id} after {frame_count - self.last_active_frame} frames")
        self.is_active = True
        self.last_active_frame = frame_count
        
        # Update state
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
        elif self.state == 'lost':
            self.state = 'confirmed'
        
        # Add to history
        self.history.append({
            'frame': self.age,
            'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox,
            'confidence': confidence
        })
    
    def update_appearance(self, feature_vector):
        """Update appearance features for re-identification"""
        if feature_vector is not None:
            self.appearance_features.append(feature_vector)
    
    def update_color_histogram(self, histogram):
        """Update color histogram for the track"""
        if self.color_histogram is None:
            self.color_histogram = histogram
        else:
            # Exponential moving average
            self.color_histogram = 0.7 * self.color_histogram + 0.3 * histogram
    
    def get_average_appearance(self):
        """Get average appearance feature for matching"""
        if len(self.appearance_features) == 0:
            return None
        return np.mean(self.appearance_features, axis=0)
    
    def _state_to_bbox(self, state):
        """Convert Kalman filter state to bbox format"""
        cx, cy, w, h = state[0:4]
        return np.array([
            cx - w/2,
            cy - h/2,
            cx + w/2,
            cy + h/2
        ])
    
    @property
    def is_activated(self):
        """Check if track is activated (confirmed or recently updated)"""
        return self.is_active and (self.state == 'confirmed' or (self.state == 'tentative' and self.hits >= 1))


class EnhancedConeTracker:
    """Enhanced tracker with ByteTrack-style two-stage matching and re-identification"""
    
    def __init__(self, 
                 track_activation_threshold=0.5,
                 high_match_threshold=0.8,
                 low_match_threshold=0.5,
                 max_time_lost=60,  # Increased for better occlusion handling
                 min_hits=3,
                 min_distance_from_football=50,
                 appearance_weight=0.3,
                 max_cones=None,
                 enforce_cone_limit=True,
                 learn_layout=False):
        
        # Thresholds
        self.track_activation_threshold = track_activation_threshold
        self.high_match_threshold = high_match_threshold
        self.low_match_threshold = low_match_threshold
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits
        self.min_distance_from_football = min_distance_from_football
        self.appearance_weight = appearance_weight
        
        # Track management
        self.tracks = []
        self.track_id_count = 0
        self.frame_count = 0
        
        # Cone count constraint
        self.max_cones = max_cones
        self.enforce_cone_limit = enforce_cone_limit
        self.total_cones_seen = 0
        self.at_cone_limit = False
        
        # Track slot system - pre-allocate IDs when max_cones is set
        self.use_slot_system = max_cones is not None and enforce_cone_limit
        if self.use_slot_system:
            print(f"Using track slot system with {max_cones} pre-allocated IDs")
            # Pre-create all track slots but leave them empty
            for i in range(max_cones):
                # We'll create actual tracks as needed, but reserve the IDs
                pass
        
        # Spatial layout learning
        self.learn_layout = learn_layout
        self.cone_positions = {}  # track_id -> average position
        self.position_history = {}  # track_id -> list of positions
        self.layout_learned = False
        self.learning_frames = 100  # Frames to learn layout
        
        # Football tracking
        self.football_history = deque(maxlen=30)
        
        # Cone size statistics
        self.cone_sizes = deque(maxlen=20)
        self.max_cone_size_variance = 0.4  # Allow 40% variance
        
        # Lost tracks buffer for re-identification
        self.lost_tracks = []
        self.lost_track_buffer = 30 if max_cones is None else float('inf')  # Keep forever if cone count known
        
    def update_football(self, football_bbox):
        """Update football position history"""
        if football_bbox is not None:
            self.football_history.append((self.frame_count, football_bbox))
    
    def extract_color_histogram(self, image, bbox):
        """Extract HSV color histogram from bbox region"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram focusing on Hue channel (more lighting invariant)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def _is_near_recent_football(self, bbox, frames_back=10):
        """Check if bbox is near any recent football position"""
        if not self.football_history:
            return False
        
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
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
        
        if len(self.cone_sizes) >= 3:
            avg_size = np.mean(self.cone_sizes)
            if abs(size - avg_size) / avg_size > self.max_cone_size_variance:
                return False
        
        return True
    
    def _update_spatial_layout(self, track_id, bbox):
        """Update spatial layout learning for a track"""
        if not self.learn_layout or self.layout_learned:
            return
        
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
        if track_id not in self.position_history:
            self.position_history[track_id] = []
        
        self.position_history[track_id].append(center)
        
        # After learning_frames, compute average positions
        if self.frame_count >= self.learning_frames and not self.layout_learned:
            self._finalize_layout_learning()
    
    def _finalize_layout_learning(self):
        """Compute average positions for all tracks"""
        for track_id, positions in self.position_history.items():
            if len(positions) > 10:  # Need sufficient samples
                self.cone_positions[track_id] = np.mean(positions, axis=0)
        
        self.layout_learned = True
        print(f"Layout learned: {len(self.cone_positions)} cone positions recorded")
    
    def _find_nearest_expected_position(self, bbox):
        """Find the track ID of the nearest expected position"""
        if not self.layout_learned or len(self.cone_positions) == 0:
            return None
        
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        min_dist = float('inf')
        nearest_track_id = None
        
        for track_id, expected_pos in self.cone_positions.items():
            dist = np.linalg.norm(center - expected_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_track_id = track_id
        
        # Only return if reasonably close (within 100 pixels)
        if min_dist < 100:
            return nearest_track_id
        return None
    
    def _comprehensive_match_to_inactive_tracks(self, bbox, confidence, frame=None):
        """
        Comprehensively try to match a detection to any inactive track
        Returns the best matching track or None
        """
        best_track = None
        best_score = float('inf')
        
        # Get all inactive tracks
        inactive_tracks = [t for t in self.tracks if not t.is_active]
        
        if len(inactive_tracks) == 0:
            return None
        
        print(f"Attempting to match detection to {len(inactive_tracks)} inactive tracks")
        
        # Calculate scores for each inactive track
        for track in inactive_tracks:
            score_components = {}
            
            # 1. IoU score with predicted position
            predicted_bbox = track.bbox  # This was updated by predict()
            iou = self._calculate_iou_single(bbox, predicted_bbox)
            score_components['iou'] = 1.0 - iou  # Convert to cost
            
            # 2. Distance to last known position
            last_center = np.array([(predicted_bbox[0] + predicted_bbox[2]) / 2,
                                   (predicted_bbox[1] + predicted_bbox[3]) / 2])
            det_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            pixel_distance = np.linalg.norm(det_center - last_center)
            score_components['distance'] = pixel_distance / 100.0  # Normalize
            
            # 3. Appearance score (if available)
            if track.color_histogram is not None and frame is not None:
                det_hist = self.extract_color_histogram(frame, bbox)
                if det_hist is not None:
                    similarity = cv2.compareHist(track.color_histogram, det_hist, cv2.HISTCMP_CORREL)
                    score_components['appearance'] = 1.0 - max(0, similarity)
                else:
                    score_components['appearance'] = 0.5
            else:
                score_components['appearance'] = 0.5
            
            # 4. Spatial layout score (if learned)
            if self.layout_learned and track.track_id in self.cone_positions:
                expected_pos = self.cone_positions[track.track_id]
                spatial_dist = np.linalg.norm(det_center - expected_pos)
                score_components['spatial'] = spatial_dist / 100.0
            else:
                score_components['spatial'] = 0.5
            
            # 5. Time penalty - prefer recently lost tracks
            frames_inactive = self.frame_count - track.last_active_frame
            score_components['time'] = min(1.0, frames_inactive / 100.0)
            
            # Calculate weighted score
            weights = {
                'iou': 0.3,
                'distance': 0.2,
                'appearance': 0.2,
                'spatial': 0.2,
                'time': 0.1
            }
            
            total_score = sum(score_components[k] * weights[k] for k in weights)
            
            print(f"  Track {track.track_id}: scores={score_components}, total={total_score:.3f}")
            
            if total_score < best_score:
                best_score = total_score
                best_track = track
        
        # Accept match if score is good enough
        threshold = 0.7 if self.at_cone_limit else 0.5
        if best_score < threshold:
            print(f"  Best match: Track {best_track.track_id} with score {best_score:.3f}")
            return best_track
        else:
            print(f"  No good match found (best score {best_score:.3f} > threshold {threshold})")
            return None
    
    def _calculate_iou_single(self, box1, box2):
        """Calculate IoU between two boxes"""
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate union
        union = area1 + area2 - intersection
        
        # Calculate IoU
        return intersection / (union + 1e-6)
    
    def update(self, detections, frame=None, appearance_features=None):
        """
        Update tracker with new detections using ByteTrack-style two-stage matching
        
        Args:
            detections: List of dicts with 'bbox' and 'confidence' keys
            frame: Current frame for color histogram extraction (optional)
            appearance_features: Dict mapping detection index to feature vectors (optional)
        """
        self.frame_count += 1
        
        # Convert detections to arrays
        if len(detections) == 0:
            det_bboxes = np.empty((0, 4))
            det_scores = np.empty(0)
        else:
            det_bboxes = np.array([d['bbox'] for d in detections])
            det_scores = np.array([d.get('confidence', 1.0) for d in detections])
        
        # Split detections into high and low confidence
        high_indices = det_scores >= self.track_activation_threshold
        low_indices = ~high_indices
        
        high_det_bboxes = det_bboxes[high_indices]
        low_det_bboxes = det_bboxes[low_indices]
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Get activated tracks (confirmed or tentative with hits)
        activated_tracks = [t for t in self.tracks if t.is_activated]
        
        # Adjust thresholds if at cone limit - be more permissive
        high_thresh = self.high_match_threshold
        low_thresh = self.low_match_threshold
        if self.at_cone_limit:
            high_thresh *= 0.7  # Relax threshold
            low_thresh *= 0.7
            print(f"At cone limit - relaxing thresholds: high={high_thresh:.2f}, low={low_thresh:.2f}")
        
        # First association: match high-confidence detections with activated tracks
        if len(high_det_bboxes) > 0 and len(activated_tracks) > 0:
            matched1, unmatched_dets1, unmatched_tracks1 = self._match_detections(
                high_det_bboxes, activated_tracks, threshold=high_thresh,
                use_appearance=True, frame=frame
            )
            
            # Update matched tracks
            for m in matched1:
                det_idx = np.where(high_indices)[0][m[0]]
                track = activated_tracks[m[1]]
                track.update_bbox(det_bboxes[det_idx], det_scores[det_idx], self.frame_count)
                
                # Update appearance features if available
                if appearance_features and det_idx in appearance_features:
                    track.update_appearance(appearance_features[det_idx])
                
                # Update color histogram if frame is available
                if frame is not None:
                    hist = self.extract_color_histogram(frame, det_bboxes[det_idx])
                    if hist is not None:
                        track.update_color_histogram(hist)
                
                # Update cone size statistics
                if track.state == 'confirmed':
                    width = det_bboxes[det_idx][2] - det_bboxes[det_idx][0]
                    height = det_bboxes[det_idx][3] - det_bboxes[det_idx][1]
                    self.cone_sizes.append(width * height)
                
                # Update spatial layout learning
                self._update_spatial_layout(track.track_id, det_bboxes[det_idx])
        else:
            matched1 = np.empty((0, 2), dtype=int)
            unmatched_dets1 = np.arange(len(high_det_bboxes))
            unmatched_tracks1 = np.arange(len(activated_tracks))
        
        # Second association: match remaining tracks with low-confidence detections
        remaining_tracks = [activated_tracks[i] for i in unmatched_tracks1]
        if len(low_det_bboxes) > 0 and len(remaining_tracks) > 0:
            matched2, unmatched_dets2, unmatched_tracks2 = self._match_detections(
                low_det_bboxes, remaining_tracks, threshold=low_thresh,
                use_appearance=True, frame=frame
            )
            
            # Update matched tracks with low-confidence detections
            for m in matched2:
                det_idx = np.where(low_indices)[0][m[0]]
                track_idx = unmatched_tracks1[m[1]]
                track = activated_tracks[track_idx]
                track.update_bbox(det_bboxes[det_idx], det_scores[det_idx], self.frame_count)
                
                # Update appearance features
                if appearance_features and det_idx in appearance_features:
                    track.update_appearance(appearance_features[det_idx])
                
                # Update color histogram
                if frame is not None:
                    hist = self.extract_color_histogram(frame, det_bboxes[det_idx])
                    if hist is not None:
                        track.update_color_histogram(hist)
        else:
            unmatched_tracks2 = list(range(len(remaining_tracks)))
        
        # Get final unmatched tracks
        unmatched_tracks_final = [remaining_tracks[i] for i in unmatched_tracks2]
        
        # Third association: try to re-identify lost tracks
        if len(self.lost_tracks) > 0 and len(high_det_bboxes) > 0:
            unmatched_high_dets = [np.where(high_indices)[0][i] for i in unmatched_dets1]
            remaining_high_bboxes = det_bboxes[unmatched_high_dets]
            
            if len(remaining_high_bboxes) > 0:
                matched3, unmatched_dets3, _ = self._match_detections(
                    remaining_high_bboxes, self.lost_tracks, threshold=0.4,
                    use_appearance=True, appearance_only=True, frame=frame
                )
                
                # Reactivate matched lost tracks
                for m in matched3:
                    det_idx = unmatched_high_dets[m[0]]
                    track = self.lost_tracks[m[1]]
                    track.update_bbox(det_bboxes[det_idx], det_scores[det_idx], self.frame_count)
                    track.state = 'confirmed'
                    self.tracks.append(track)
                    
                    # Update appearance
                    if appearance_features and det_idx in appearance_features:
                        track.update_appearance(appearance_features[det_idx])
                    
                    if frame is not None:
                        hist = self.extract_color_histogram(frame, det_bboxes[det_idx])
                        if hist is not None:
                            track.update_color_histogram(hist)
                
                # Remove reactivated tracks from lost tracks
                self.lost_tracks = [self.lost_tracks[i] for i in range(len(self.lost_tracks)) 
                                  if i not in [m[1] for m in matched3]]
                
                # Update unmatched detections
                unmatched_dets1 = [unmatched_dets1[i] for i in unmatched_dets3]
        
        # Check if we're at cone limit
        if self.max_cones is not None:
            self.at_cone_limit = self.total_cones_seen >= self.max_cones
        
        # Fourth association: AGGRESSIVE re-identification when at cone limit
        # Try ALL remaining detections (both high and low) with ALL lost tracks
        if self.at_cone_limit and len(self.lost_tracks) > 0:
            # Collect all unmatched detections
            all_unmatched_indices = []
            
            # Add remaining high confidence detections
            for i in unmatched_dets1:
                all_unmatched_indices.append(np.where(high_indices)[0][i])
            
            # Add ALL low confidence detections that weren't matched
            if 'unmatched_dets2' in locals():
                for i in unmatched_dets2:
                    all_unmatched_indices.append(np.where(low_indices)[0][i])
            else:
                # If no second matching happened, add all low conf detections
                for i in range(len(low_det_bboxes)):
                    all_unmatched_indices.append(np.where(low_indices)[0][i])
            
            if len(all_unmatched_indices) > 0:
                all_unmatched_bboxes = det_bboxes[all_unmatched_indices]
                
                print(f"At cone limit: Aggressively trying to match {len(all_unmatched_indices)} detections with {len(self.lost_tracks)} lost tracks")
                
                # Try matching with very relaxed threshold
                matched4, unmatched_dets4, _ = self._match_detections(
                    all_unmatched_bboxes, self.lost_tracks, threshold=0.3,
                    use_appearance=True, frame=frame
                )
                
                # Reactivate matched lost tracks
                reactivated_count = 0
                for m in matched4:
                    det_idx = all_unmatched_indices[m[0]]
                    track = self.lost_tracks[m[1]]
                    track.update_bbox(det_bboxes[det_idx], det_scores[det_idx], self.frame_count)
                    track.state = 'confirmed'
                    self.tracks.append(track)
                    reactivated_count += 1
                    
                    # Update appearance
                    if appearance_features and det_idx in appearance_features:
                        track.update_appearance(appearance_features[det_idx])
                    
                    if frame is not None:
                        hist = self.extract_color_histogram(frame, det_bboxes[det_idx])
                        if hist is not None:
                            track.update_color_histogram(hist)
                
                if reactivated_count > 0:
                    print(f"Successfully reactivated {reactivated_count} lost tracks")
                    # Remove reactivated tracks from lost tracks
                    self.lost_tracks = [self.lost_tracks[i] for i in range(len(self.lost_tracks)) 
                                      if i not in [m[1] for m in matched4]]
                    
                    # Update unmatched detection lists
                    matched_det_indices = [all_unmatched_indices[m[0]] for m in matched4]
                    unmatched_dets1 = [i for i in unmatched_dets1 
                                      if np.where(high_indices)[0][i] not in matched_det_indices]
        
        # Create new tracks for unmatched high-confidence detections
        for i in unmatched_dets1:
            det_idx = np.where(high_indices)[0][i]
            bbox = det_bboxes[det_idx]
            
            # Validate before creating new track
            if not self._is_near_recent_football(bbox) and self._validate_cone_size(bbox):
                # Check cone limit constraint
                if self.enforce_cone_limit and self.at_cone_limit:
                    # Don't create new track - use comprehensive matching
                    print(f"At cone limit ({self.max_cones}), attempting comprehensive re-identification")
                    
                    # Try comprehensive matching with all inactive tracks
                    matched_track = self._comprehensive_match_to_inactive_tracks(
                        bbox, det_scores[det_idx], frame
                    )
                    
                    if matched_track:
                        # Reactivate the matched track
                        matched_track.update_bbox(bbox, det_scores[det_idx], self.frame_count)
                        
                        # Update appearance
                        if appearance_features and det_idx in appearance_features:
                            matched_track.update_appearance(appearance_features[det_idx])
                        
                        if frame is not None:
                            hist = self.extract_color_histogram(frame, bbox)
                            if hist is not None:
                                matched_track.update_color_histogram(hist)
                        
                        # Update spatial layout
                        self._update_spatial_layout(matched_track.track_id, bbox)
                    else:
                        print(f"Failed to match detection to any inactive track - detection will be ignored")
                    
                    continue  # Skip creating new track
                
                # Create new track if not at limit
                if self.use_slot_system and self.total_cones_seen < self.max_cones:
                    # Use slot ID instead of incrementing counter
                    slot_id = self.total_cones_seen
                    track = ConeTrack(slot_id, bbox, det_scores[det_idx])
                    self.total_cones_seen += 1
                    print(f"Creating new track with slot ID {slot_id}")
                else:
                    # Original behavior
                    track = ConeTrack(self.track_id_count, bbox, det_scores[det_idx])
                    self.track_id_count += 1
                    self.total_cones_seen += 1
                
                self.tracks.append(track)
                
                # Initialize appearance
                if appearance_features and det_idx in appearance_features:
                    track.update_appearance(appearance_features[det_idx])
                
                if frame is not None:
                    hist = self.extract_color_histogram(frame, bbox)
                    if hist is not None:
                        track.update_color_histogram(hist)
        
        # Mark tracks as inactive instead of removing them
        if self.use_slot_system:
            # With slot system, never remove tracks, just mark them inactive
            for track in self.tracks:
                if track.time_since_update > self.max_time_lost and track.is_active:
                    track.is_active = False
                    track.last_active_frame = self.frame_count
                    print(f"Marking track {track.track_id} as inactive (time_since_update: {track.time_since_update}, last seen: frame {track.last_active_frame})")
        else:
            # Original behavior when not using slot system
            # Move lost tracks to lost buffer
            for track in self.tracks:
                if track.time_since_update > self.max_time_lost:
                    if track.state == 'confirmed':
                        # Always keep lost tracks when at cone limit
                        if self.at_cone_limit or len(track.appearance_features) > 0 or track.color_histogram is not None:
                            self.lost_tracks.append(track)
                            print(f"Moving track {track.track_id} to lost buffer (age: {track.age}, hits: {track.hits})")
            
            # Remove old lost tracks
            self.lost_tracks = [t for t in self.lost_tracks 
                              if self.frame_count - t.age < self.lost_track_buffer]
            
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_time_lost]
        
        # Return active tracks
        results = []
        for track in self.tracks:
            if track.is_activated:
                results.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'age': track.age,
                    'confirmed': track.state == 'confirmed',
                    'state': track.state
                })
        
        return results
    
    def _match_detections(self, detections, tracks, threshold=0.8, use_appearance=False, appearance_only=False, frame=None):
        """
        Match detections to tracks using IoU and optionally appearance features
        """
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), []
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], np.arange(len(tracks))
        
        # Calculate cost matrix
        if appearance_only and use_appearance:
            # Use only appearance distance
            cost_matrix = self._calculate_appearance_cost(detections, tracks, frame)
        else:
            # Calculate IoU matrix
            iou_matrix = self._calculate_iou_matrix(
                detections,
                np.array([t.bbox for t in tracks])
            )
            
            if use_appearance:
                # Combine IoU and appearance costs
                appearance_cost = self._calculate_appearance_cost(detections, tracks, frame)
                # Cost = weighted combination (lower is better)
                cost_matrix = (1 - self.appearance_weight) * (1 - iou_matrix) + \
                             self.appearance_weight * appearance_cost
            else:
                cost_matrix = 1 - iou_matrix
        
        # Solve assignment problem
        if min(cost_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_indices = np.column_stack((row_ind, col_ind))
            
            # Filter matches based on threshold
            matches = []
            for m in matched_indices:
                if appearance_only:
                    if cost_matrix[m[0], m[1]] < (1 - threshold):
                        matches.append(m)
                else:
                    iou = 1 - cost_matrix[m[0], m[1]] if not use_appearance else \
                          (1 - cost_matrix[m[0], m[1]]) / (1 - self.appearance_weight)
                    if iou >= threshold:
                        matches.append(m)
            
            matches = np.array(matches) if matches else np.empty((0, 2), dtype=int)
        else:
            matches = np.empty((0, 2), dtype=int)
        
        # Find unmatched detections and tracks
        unmatched_detections = []
        for d in range(len(detections)):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_tracks = []
        for t in range(len(tracks)):
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
        
        # Return as 2D matrix
        if iou.ndim == 3:
            return iou.squeeze(axis=0)
        return iou
    
    def _calculate_appearance_cost(self, detections, tracks, frame=None):
        """Calculate appearance-based cost matrix using color histograms"""
        n_dets = len(detections)
        n_tracks = len(tracks)
        cost_matrix = np.ones((n_dets, n_tracks))
        
        # If no frame provided, can't calculate histograms
        if frame is None:
            return cost_matrix
        
        # Calculate histogram for each detection
        det_histograms = []
        for det_bbox in detections:
            hist = self.extract_color_histogram(frame, det_bbox)
            det_histograms.append(hist)
        
        # Compare with track histograms
        for t_idx, track in enumerate(tracks):
            if track.color_histogram is not None:
                for d_idx, det_hist in enumerate(det_histograms):
                    if det_hist is not None:
                        # Calculate histogram similarity (correlation)
                        similarity = cv2.compareHist(track.color_histogram, det_hist, cv2.HISTCMP_CORREL)
                        # Convert to cost (0 = perfect match, 1 = no match)
                        cost_matrix[d_idx, t_idx] = 1.0 - max(0, similarity)
        
        return cost_matrix
    
    def get_tracks_summary(self):
        """Get summary of all active tracks"""
        return {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': sum(1 for t in self.tracks if t.state == 'confirmed'),
            'tentative_tracks': sum(1 for t in self.tracks if t.state == 'tentative'),
            'lost_tracks': sum(1 for t in self.tracks if t.state == 'lost'),
            'lost_buffer_tracks': len(self.lost_tracks),
            'frame_count': self.frame_count
        }
    
    def save_tracks(self, filename):
        """Save tracking history to JSON file"""
        data = {
            'frame_count': self.frame_count,
            'tracks': []
        }
        
        all_tracks = self.tracks + self.lost_tracks
        for track in all_tracks:
            data['tracks'].append({
                'track_id': track.track_id,
                'history': list(track.history),
                'final_confidence': track.confidence,
                'total_age': track.age,
                'state': track.state,
                'hits': track.hits
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)