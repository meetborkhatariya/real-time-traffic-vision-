import cv2
import time
import os
from ultralytics import YOLO

class TrafficVisionSystem:
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize the Traffic Vision System with a YOLO model."""
        self.model = YOLO(model_path)
        self.vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

    def process_image(self, image_source):
        """Perform object detection on a single image."""
        results = self.model(image_source, classes=self.vehicle_classes)
        vehicle_count = len(results[0].boxes)
        print(f"Total Vehicles detected: {vehicle_count}")
        
        # Display/Save result
        annotated_img = results[0].plot()
        return annotated_img, vehicle_count

    def process_video(self, video_path, output_path="output.mp4"):
        """Track vehicles and count them crossing a line in a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps_in if fps_in > 0 else 20, 
                              (width, height))

        prev_time = time.time()
        line_y = height // 2
        crossed_ids = set()
        total_crossed = 0

        print(f"Processing video: {video_path}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Tracking with object persistence
            results = self.model.track(frame, classes=self.vehicle_classes, persist=True)
            boxes = results[0].boxes
            vehicle_count = len(boxes)

            # Density Analysis
            if vehicle_count <= 2:
                density = "LOW"
            elif vehicle_count <= 5:
                density = "MEDIUM"
            else:
                density = "HIGH"

            annotated_frame = results[0].plot()

            # Line crossing logic using Track IDs
            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes.xyxy, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cy = int((y1 + y2) / 2)

                    # Count only once per ID when it crosses the line
                    if cy > line_y and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        total_crossed += 1

            # UI Overlays
            cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(annotated_frame,
                        f"Vehicles: {vehicle_count} | Density: {density} | FPS: {int(fps)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(annotated_frame,
                        f"Total Crossed: {total_crossed}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"Processing complete. Saved to {output_path}")
        return total_crossed

if __name__ == "__main__":
    tvs = TrafficVisionSystem()
    
    # Example usage for video
    # If traffic.mp4 doesn't exist, you might want to provide a path or download one
    video_file = "traffic.mp4"
    if os.path.exists(video_file):
        tvs.process_video(video_file)
    else:
        print(f"Please provide a '{video_file}' file to run the video analysis.")
        # Example for image if you want to test
        # tvs.process_image("https://ultralytics.com/images/bus.jpg")
