from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from sqlalchemy.orm import Session
from sqlalchemy import func
import base64

from database import SessionLocal, TrafficEvent
from vision_core import VisionService

# Global vision service instance
_vision_service = None

@asynccontextmanager
async def lifespan(app):
    """Fast startup: model loading is now lazy to avoid Render timeouts."""
    print("🚀 Server starting (Fast Mode)...")
    yield
    print("Server shutting down.")

app = FastAPI(title="VisionFlow AI Backend", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_vision_service():
    global _vision_service
    if _vision_service is None:
        print("Model not loaded yet — loading now...")
        try:
            _vision_service = VisionService(model_path="yolov8n.pt")
        except Exception as e:
            print(f"FATAL: Failed to initialize VisionService: {e}")
            raise e
    return _vision_service

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

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/")
def read_root():
    try:
        service = get_vision_service()
        return {"status": "online", "model": service.current_model_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/config/model")
def update_model(config: ModelConfig):
    """Switch YOLO model version asynchronously."""
    service = get_vision_service()
    service.switch_model(config.model_name)
    return {"status": "success", "new_model": service.current_model_name}

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
async def process_image(file: UploadFile = File(...), conf: float = 0.35, db: Session = Depends(get_db)):
    """Process a single image and return base64 string."""
    import cv2
    import numpy as np
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image data"})

        print(f"Processing image with conf={conf}")
        service = get_vision_service()
        annotated_img, count, types = service.process_image(img, conf_threshold=conf)
        
        # Save to database
        if types:
            for v_type in types:
                db_event = TrafficEvent(vehicle_type=v_type, track_id=0, direction="static_image")
                db.add(db_event)
            db.commit()
        
        # Encode to base64 with compression
        _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img_str = base64.b64encode(buffer).decode("utf-8")
        
        return {"count": count, "image": img_str}
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/video/stream")
async def process_video_stream(file: UploadFile = File(...), conf: float = 0.35):
    """Accepts a video file upload and returns a real-time MJPEG stream."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        content = await file.read()
        tfile.write(content)
        tfile.close() 
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {e}"})
    
    def generate_frames():
        import cv2
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            print(f"Failed to open video file: {tfile.name}")
            yield b""
            return
            
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = height // 2
        crossed_ids = set()
        track_history = {}
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                    
                service = get_vision_service()
                annotated_frame, curr_count, density, new_events = service.process_frame(frame, line_y, crossed_ids, track_history, conf_threshold=conf)
                
                if new_events:
                    db = SessionLocal()
                    try:
                        for event in new_events:
                            db_event = TrafficEvent(vehicle_type=event["vehicle_type"], track_id=event["track_id"], direction="crossing")
                            db.add(db_event)
                        db.commit()
                    except Exception as db_e:
                        print(f"Database error in stream: {db_e}")
                    finally:
                        db.close()
                    
                cv2.putText(annotated_frame, f"Density: {density} | Session Crossed: {len(crossed_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret: continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
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
        import cv2
        print(f"Connecting to URL: {payload.url}")
        cap = cv2.VideoCapture(payload.url)
        if not cap.isOpened():
            print(f"Failed to open video URL: {payload.url}")
            yield b""
            return
        
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = height // 2
        crossed_ids = set()
        track_history = {}
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                service = get_vision_service()
                annotated_frame, curr_count, density, new_events = service.process_frame(frame, line_y, crossed_ids, track_history, conf_threshold=payload.conf)
                if new_events:
                    db = SessionLocal()
                    try:
                        for event in new_events:
                            db_event = TrafficEvent(vehicle_type=event["vehicle_type"], track_id=event["track_id"], direction="crossing")
                            db.add(db_event)
                        db.commit()
                    except Exception as e:
                        print(f"DB Error in URL stream: {e}")
                    finally:
                        db.close()
                
                cv2.putText(annotated_frame, f"Density: {density} | Session Crossed: {len(crossed_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret: continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/video/webcam")
async def process_webcam(conf: float = 0.35):
    """Accesses local webcam 0 mapping from the backend layer."""
    def generate_frames():
        import cv2
        print("Opening backend webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam 0 not available on backend server.")
            yield b""
            return
            
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = height // 2
        crossed_ids = set()
        track_history = {}
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                service = get_vision_service()
                annotated_frame, curr_count, density, new_events = service.process_frame(frame, line_y, crossed_ids, track_history, conf_threshold=conf)
                if new_events:
                    db = SessionLocal()
                    try:
                        for event in new_events:
                            db_event = TrafficEvent(vehicle_type=event["vehicle_type"], track_id=event["track_id"], direction="crossing")
                            db.add(db_event)
                        db.commit()
                    except Exception as e:
                        print(f"DB Error in Webcam stream: {e}")
                    finally:
                        db.close()
                        
                cv2.putText(annotated_frame, f"Density: {density} | Session Crossed: {len(crossed_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret: continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
