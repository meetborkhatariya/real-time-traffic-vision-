import cv2
from ultralytics import YOLO
import os

class VisionService:
    def __init__(self, model_path="yolov8n.pt"):
        self.current_model_name = model_path
        self._load_model(model_path)
        self.vehicle_classes = [1, 2, 3, 5, 7] # bicycle, car, motorcycle, bus, truck

    def _load_model(self, model_path):
        # Assumes models might be downloaded locally or uses YOLO ultralytics defaults
        try:
            self.model = YOLO(model_path)
            print(f"Loaded model {model_path} successfully.")
        except Exception as e:
            print(f"Failed to load {model_path}, falling back to yolov8n.pt")
            self.model = YOLO("yolov8n.pt")

    def switch_model(self, model_name):
        if model_name != self.current_model_name:
            self._load_model(model_name)
            self.current_model_name = model_name

    def process_image(self, img_np, conf_threshold=0.35):
        """Processes a single static image."""
        results = self.model(img_np, conf=conf_threshold, classes=self.vehicle_classes)
        annotated_img = results[0].plot()
        boxes = results[0].boxes
        count = len(boxes)
        types = []
        if boxes.cls is not None:
             classes = boxes.cls.int().cpu().tolist()
             types = [self.model.names[c] for c in classes]
        return annotated_img, count, types

    def process_frame(self, frame, line_y, crossed_ids, track_history, conf_threshold=0.35):
        """Processes video frame for tracking using Vector State-Machine."""
        results = self.model.track(frame, classes=self.vehicle_classes, persist=True, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        curr_count = len(boxes)
        
        density = "LOW" if curr_count <= 2 else "MEDIUM" if curr_count <= 6 else "HIGH"
        new_crossed_events = []

        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls_id in zip(boxes.xyxy, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cy = int((y1 + y2) / 2)
                
                # Vector-based State Machine Logic
                if track_id in track_history:
                    prev_cy = track_history[track_id]
                    # Only count if it actively moved from ABOVE the line to BELOW the line
                    if prev_cy <= line_y and cy > line_y and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        vehicle_type = self.model.names[cls_id]
                        new_crossed_events.append({
                            "track_id": track_id, 
                            "vehicle_type": vehicle_type
                        })
                
                # Save the current center-point for the next frame
                track_history[track_id] = cy

        annotated_frame = results[0].plot()
        cv2.line(annotated_frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)
        
        return annotated_frame, curr_count, density, new_crossed_events
