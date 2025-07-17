import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Visualizer:
    def __init__(self, speed_thresholds=(5, 15, 25)):
        # Speed thresholds for color mapping (pixels/frame)
        self.low_speed, self.med_speed, self.high_speed = speed_thresholds
        
        # Color map for speed visualization
        self.colormap = cm.get_cmap('RdYlGn_r')  # Red-Yellow-Green reversed
        
        # Store football paths with their speeds
        self.football_paths = defaultdict(list)  # track_id -> [(point, speed), ...]
        
        # Colors for bounding boxes (different for each class)
        self.bbox_colors = {
            0: (0, 255, 0),    # Green for football
            1: (255, 0, 0)     # Blue for cone
        }
        
    def draw_tracks(self, frame, tracks):
        # Draw bounding boxes and IDs for all tracks
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            class_id = track['class_id']
            class_name = track['class_name']
            
            # Get bbox coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = self.bbox_colors.get(class_id, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and class name
            label = f"ID:{track_id} {class_name}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(frame, 
                         (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label,
                       (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
            
            # Update football path if this is a football
            if class_id == 0 and len(track.get('history', [])) > 1:
                self._update_football_path(track_id, track['history'])
        
        # Draw football paths with speed colors
        self._draw_football_paths(frame)
        
        return frame
    
    def _update_football_path(self, track_id, history):
        # Clear existing path for this track
        self.football_paths[track_id] = []
        
        # Calculate speed between consecutive points
        for i in range(1, len(history)):
            prev_bbox = history[i-1]
            curr_bbox = history[i]
            
            # Calculate centers
            prev_center = self._get_center(prev_bbox)
            curr_center = self._get_center(curr_bbox)
            
            # Calculate speed (pixels per frame)
            speed = np.linalg.norm(curr_center - prev_center)
            
            # Store point with speed
            self.football_paths[track_id].append((curr_center, speed))
    
    def _draw_football_paths(self, frame):
        for track_id, path_data in self.football_paths.items():
            if len(path_data) < 2:
                continue
            
            # Draw path segments with color based on speed
            for i in range(1, len(path_data)):
                prev_point, _ = path_data[i-1]
                curr_point, speed = path_data[i]
                
                # Get color based on speed
                color = self._speed_to_color(speed)
                
                # Draw line segment
                cv2.line(frame,
                        tuple(prev_point.astype(int)),
                        tuple(curr_point.astype(int)),
                        color, 3)
                
                # Draw points for better visibility
                cv2.circle(frame, tuple(curr_point.astype(int)), 2, color, -1)
    
    def _get_center(self, bbox):
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    def _speed_to_color(self, speed):
        # Normalize speed to 0-1 range
        if speed <= self.low_speed:
            normalized = 0.0
        elif speed >= self.high_speed:
            normalized = 1.0
        else:
            normalized = (speed - self.low_speed) / (self.high_speed - self.low_speed)
        
        # Get color from colormap
        rgba = self.colormap(normalized)
        # Convert to BGR for OpenCV
        bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        
        return bgr
    
    def add_speed_legend(self, frame):
        # Add a speed color legend to the frame
        legend_height = 20
        legend_width = 200
        legend_x = frame.shape[1] - legend_width - 20
        legend_y = 20
        
        # Create gradient
        for i in range(legend_width):
            normalized = i / legend_width
            color = self._speed_to_color(normalized * self.high_speed)
            cv2.line(frame,
                    (legend_x + i, legend_y),
                    (legend_x + i, legend_y + legend_height),
                    color, 1)
        
        # Add border
        cv2.rectangle(frame,
                     (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (255, 255, 255), 1)
        
        # Add labels
        cv2.putText(frame, "Low", (legend_x - 30, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "High", (legend_x + legend_width + 5, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Speed", (legend_x + legend_width//2 - 20, legend_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame