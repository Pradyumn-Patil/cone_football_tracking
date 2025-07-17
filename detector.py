import numpy as np
import cv2
import onnxruntime as ort


class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape from model
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Class names
        self.class_names = {0: 'football', 1: 'cone'}
    
    def preprocess(self, image):
        # Store original dimensions
        self.orig_height, self.orig_width = image.shape[:2]
        
        # Resize image to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        transposed = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def postprocess(self, outputs):
        # Extract predictions
        predictions = outputs[0][0]  # Remove batch dimension
        
        detections = []
        
        # Process each detection
        for pred in predictions.T:
            # Extract confidence scores for each class
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter by confidence threshold
            if confidence < self.conf_threshold:
                continue
            
            # Extract bounding box coordinates
            cx, cy, w, h = pred[:4]
            
            # Convert from center coordinates to corner coordinates
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # Scale coordinates back to original image size
            x1 = int(x1 * self.orig_width / self.input_width)
            y1 = int(y1 * self.orig_height / self.input_height)
            x2 = int(x2 * self.orig_width / self.input_width)
            y2 = int(y2 * self.orig_height / self.input_height)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.orig_width - 1, x2)
            y2 = min(self.orig_height - 1, y2)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': self.class_names[class_id]
            })
        
        return detections
    
    def detect(self, image):
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess results
        detections = self.postprocess(outputs)
        
        return detections