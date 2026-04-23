from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import tempfile
from sqlalchemy.orm import Session
from sqlalchemy import func
import os
import numpy as np
import base64

from database import SessionLocal, TrafficEvent
from vision_core import VisionService

app = FastAPI(title="VisionFlow AI Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vision_service = VisionService(model_path="yolov8n.pt") 

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ModelConfig(BaseModel):
    model_name: str
    
class VideoUrl(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"status": "online", "model": vision_service.current_model_name}

@app.post("/api/config/model")
def update_model(config: ModelConfig):
    """Switch YOLO model version asynchronously."""
    vision_service.switch_model(config.model_name)
    return {"status": "success", "new_model": vision_service.current_model_name}

@app.get("/api/analytics/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    total_crossed = db.query(TrafficEvent).count()
    breakdown = db.query(TrafficEvent.vehicle_type, func.count(TrafficEvent.id)).group_by(TrafficEvent.vehicle_type).all()
    breakdown_dict = {item[0]: item[1] for item in breakdown}
    return {"total_crossed": total_crossed, "breakdown": breakdown_dict}

@app.get("/api/analytics/data")
def get_analytics_data(db: Session = Depends(get_db)):
    """Fetch the raw database rows."""
    events = db.query(TrafficEvent).order_by(TrafficEvent.id.desc()).limit(50).all()
    return [
        {
            "ID": e.id, 
            "Time": e.timestamp.isoformat(), 
            "Type": e.vehicle_type, 
            "Source": e.direction
        } 
        for e in events
    ]

@app.post("/api/image/process")
async def process_image(file: UploadFile = File(...), conf: float = 0.35):
    """Process a single image and return base64 string."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    annotated_img, count, types = vision_service.process_image(img, conf_threshold=conf)
    
    # Save to database
    if types:
        db = SessionLocal()
        for v_type in types:
            db_event = TrafficEvent(vehicle_type=v_type, track_id=0, direction="static_image")
            db.add(db_event)
        db.commit()
        db.close()
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_str = base64.b64encode(buffer).decode("utf-8")
    
    return {"count": count, "image": img_str}

@app.post("/api/video/stream")
async def process_video_stream(file: UploadFile = File(...), conf: float = 0.35):
    """Accepts a video file upload and returns a real-time MJPEG stream."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    content = await file.read()
    tfile.write(content)
    tfile.close() 
    
    def generate_frames():
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            yield b""
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = height // 2
        crossed_ids = set()
        track_history = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            annotated_frame, curr_count, density, new_events = vision_service.process_frame(frame, line_y, crossed_ids, track_history, conf_threshold=conf)
            
            if new_events:
                db = SessionLocal()
                for event in new_events:
                    db_event = TrafficEvent(vehicle_type=event["vehicle_type"], track_id=event["track_id"], direction="crossing")
                    db.add(db_event)
                db.commit()
                db.close()
                
            # Add overlay text for the video stream
            cv2.putText(annotated_frame, f"Density: {density} | Session Crossed: {len(crossed_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        cap.release()
        try: os.remove(tfile.name)
        except: pass

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

class VideoUrl(BaseModel):
    url: str
    conf: float = 0.35

@app.post("/api/video/stream_url")
async def process_video_url(payload: VideoUrl):
    def generate_frames():
        cap = cv2.VideoCapture(payload.url)
        if not cap.isOpened():
            yield b""
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = height // 2
        crossed_ids = set()
        track_history = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            annotated_frame, curr_count, density, new_events = vision_service.process_frame(frame, line_y, crossed_ids, track_history, conf_threshold=payload.conf)
            if new_events:
                db = SessionLocal()
                for event in new_events:
                    db_event = TrafficEvent(vehicle_type=event["vehicle_type"], track_id=event["track_id"], direction="crossing")
                    db.add(db_event)
                db.commit()
                db.close()
            
            cv2.putText(annotated_frame, f"Density: {density} | Session Crossed: {len(crossed_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/video/webcam")
async def process_webcam(conf: float = 0.35):
    """Accesses local webcam 0 mapping from the backend layer."""
    def generate_frames():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield b""
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = height // 2
        crossed_ids = set()
        track_history = {}
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                annotated_frame, curr_count, density, new_events = vision_service.process_frame(frame, line_y, crossed_ids, track_history, conf_threshold=conf)
                if new_events:
                    db = SessionLocal()
                    for event in new_events:
                        db_event = TrafficEvent(vehicle_type=event["vehicle_type"], track_id=event["track_id"], direction="crossing")
                        db.add(db_event)
                    db.commit()
                    db.close()
                    
                cv2.putText(annotated_frame, f"Density: {density} | Session Crossed: {len(crossed_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret: continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
