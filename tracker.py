import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class Track:
    def __init__(self, track_id, bbox, class_id, class_name):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history = [bbox]
        self.velocity = np.zeros(4)
        
    def predict(self):
        # Simple constant velocity model
        self.bbox = self.bbox + self.velocity
        self.age += 1
        self.time_since_update += 1
        
    def update(self, bbox):
        # Update velocity
        self.velocity = bbox - self.bbox
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox.copy())
        
        # Keep only last 100 positions for history
        if len(self.history) > 100:
            self.history.pop(0)
    
    @property
    def center(self):
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        ])


class ByteTracker:
    def __init__(self, 
                 track_thresh=0.5,
                 match_thresh=0.8,
                 max_time_lost=30,
                 min_hits=10):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits
        self.tracks = []
        self.track_id_count = 0
        
    def update(self, detections):
        # Convert detections to numpy array
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']] 
                            for d in detections])
        
        # Separate high and low confidence detections
        remain_inds = dets[:, 4] > self.track_thresh
        dets_high = dets[remain_inds]
        dets_low = dets[~remain_inds]
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match high confidence detections with tracks
        matched, unmatched_dets, unmatched_tracks = self._match(dets_high, self.tracks)
        
        # Update matched tracks
        for m in matched:
            self.tracks[m[1]].update(dets_high[m[0], :4])
            if len(detections) > m[0]:
                det_idx = np.where(remain_inds)[0][m[0]]
                self.tracks[m[1]].class_id = detections[det_idx]['class_id']
                self.tracks[m[1]].class_name = detections[det_idx]['class_name']
        
        # Match remaining tracks with low confidence detections
        if len(dets_low) > 0 and len(unmatched_tracks) > 0:
            remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
            matched_low, unmatched_dets_low, unmatched_tracks_low = self._match(dets_low, remaining_tracks)
            
            # Update matched tracks with low confidence detections
            for m in matched_low:
                track_idx = unmatched_tracks[m[1]]
                self.tracks[track_idx].update(dets_low[m[0], :4])
                if len(detections) > len(dets_high) + m[0]:
                    det_idx = np.where(~remain_inds)[0][m[0]]
                    self.tracks[track_idx].class_id = detections[det_idx]['class_id']
                    self.tracks[track_idx].class_name = detections[det_idx]['class_name']
            
            # Update unmatched tracks list
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_low]
        
        # Create new tracks for unmatched high confidence detections
        for i in unmatched_dets:
            if i < len(detections):
                det_idx = np.where(remain_inds)[0][i]
                det = detections[det_idx]
                track = Track(
                    self.track_id_count,
                    np.array(det['bbox']),
                    det['class_id'],
                    det['class_name']
                )
                self.track_id_count += 1
                self.tracks.append(track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_time_lost]
        
        # Filter out short-lived tracks (likely false positives)
        active_tracks = []
        for track in self.tracks:
            if track.time_since_update < 1 or track.hits >= self.min_hits:
                active_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox.tolist(),
                    'class_id': track.class_id,
                    'class_name': track.class_name,
                    'history': track.history.copy() if track.class_id == 0 else []  # Only store history for football
                })
        
        return active_tracks
    
    def _match(self, detections, tracks):
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), []
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], np.arange(len(tracks))
        
        # Calculate IoU distance matrix
        iou_matrix = self._iou_batch(detections[:, :4], np.array([t.bbox for t in tracks]))
        
        # Solve assignment problem
        cost_matrix = 1 - iou_matrix
        
        # Use scipy's linear_sum_assignment instead of lap
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Find matched and unmatched
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for i in range(len(row_ind)):
            if cost_matrix[row_ind[i], col_ind[i]] <= self.match_thresh:
                matched.append([row_ind[i], col_ind[i]])
                unmatched_dets.remove(row_ind[i])
                unmatched_tracks.remove(col_ind[i])
        
        # Filter out matched with low IoU
        matches = np.array(matched)
        if len(matches) > 0:
            for m in matches:
                if iou_matrix[m[0], m[1]] < 1 - self.match_thresh:
                    unmatched_dets.append(m[0])
                    unmatched_tracks.append(m[1])
            
            matched = [m for m in matches if iou_matrix[m[0], m[1]] >= 1 - self.match_thresh]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _iou_batch(self, bboxes1, bboxes2):
        # Calculate IoU between two sets of bboxes
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        
        # Calculate intersection
        xA = np.maximum(x11, x21.T)
        yA = np.maximum(y11, y21.T)
        xB = np.minimum(x12, x22.T)
        yB = np.minimum(y12, y22.T)
        
        inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        
        # Calculate union
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union_area = area1 + area2.T - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou