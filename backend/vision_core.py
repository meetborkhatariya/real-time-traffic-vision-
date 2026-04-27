import os
import gc

class VisionService:
    def __init__(self, model_path="yolov8n.pt"):
        # Import heavy libraries inside __init__ to avoid blocking main API startup
        import cv2
        from ultralytics import YOLO
        import torch
        self.cv2 = cv2
        self.YOLO = YOLO
        self.torch = torch
        
        self.current_model_name = model_path
        self._load_model(model_path)
        self.vehicle_classes = [1, 2, 3, 5, 7] # bicycle, car, motorcycle, bus, truck

    def _load_model(self, model_path):
        try:
            # Clear old model from memory before loading new one
            if hasattr(self, 'model'):
                del self.model
            gc.collect()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

            print(f"Loading {model_path}...")
            self.model = self.YOLO(model_path)
            self.model.to('cpu') 
            print(f"Loaded model {model_path} successfully on CPU.")
        except Exception as e:
            print(f"Failed to load {model_path}, falling back to yolov8n.pt: {e}")
            self.model = self.YOLO("yolov8n.pt")
            self.model.to('cpu')

    def switch_model(self, model_name):
        if model_name != self.current_model_name:
            print(f"Switching model: {self.current_model_name} -> {model_name}")
            self._load_model(model_name)
            self.current_model_name = model_name
            gc.collect()

    def process_image(self, img_np, conf_threshold=0.35):
        """Processes a single static image with downscaling optimization."""
        print("Starting process_image...")
        # 1. Downscale if too large to save latency and memory
        h, w = img_np.shape[:2]
        max_dim = 1080
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_np = self.cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=self.cv2.INTER_AREA)
            print(f"Resized image to {img_np.shape}")

        # 2. Run inference
        print(f"Running inference with conf={conf_threshold}...")
        results = self.model(img_np, conf=conf_threshold, classes=self.vehicle_classes, device='cpu', verbose=False)
        print("Inference finished.")
        
        # 3. Generate annotated image
        annotated_img = results[0].plot()
        
        boxes = results[0].boxes
        count = len(boxes)
        types = []
        if boxes.cls is not None:
             classes = boxes.cls.int().cpu().tolist()
             types = [self.model.names[c] for c in classes]
             
        return annotated_img, count, types

    def process_frame(self, frame, line_y, crossed_ids, track_history, conf_threshold=0.35):
        """Processes video frame for tracking with downscaling optimization."""
        # Downscale for faster processing on CPU
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = self.cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=self.cv2.INTER_AREA)
            line_y = int(line_y * scale)

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
        self.cv2.line(annotated_frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)
        
        return annotated_frame, curr_count, density, new_crossed_events
