import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Set Page Config
st.set_page_config(
    page_title="VisionFlow AI | Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top left, #1a1c2c, #0e1117);
    }
    
    .main-header {
        background: linear-gradient(90deg, #00d4ff, #0080ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 0px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .sidebar .sidebar-content {
        background-image: linear-gradient(#161b22, #161b22);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #00d4ff, #0080ff);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model(model_name):
    weights = "yolov8n.pt" if "n" in model_name else "yolov8s.pt"
    return YOLO(weights)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965214.png", width=100)
    st.title("VisionFlow AI")
    st.markdown("---")
    
    st.subheader("⚙️ Model Settings")
    model_type = st.selectbox("Intelligence Level", [
        "Fast (YOLOv8n)", 
        "Balanced (YOLOv8s)",
        "Precise (YOLOv8m)",
        "Ultra Precise (YOLOv8l)"
    ])
    
    # Map selection to model file
    model_map = {
        "Fast (YOLOv8n)": "yolov8n.pt",
        "Balanced (YOLOv8s)": "yolov8s.pt",
        "Precise (YOLOv8m)": "yolov8m.pt",
        "Ultra Precise (YOLOv8l)": "yolov8l.pt"
    }
    
    model = load_model(model_map[model_type])
    
    conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.35)
    st.markdown("---")
    
    st.subheader("📊 Session Statistics")
    st.info("System Status: Online & Secured")

# --- UI Layout ---
st.markdown('<p class="main-header">VisionFlow Traffic Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p style="color: #8b949e; font-size: 1.2rem; margin-top: -10px;">Next-generation real-time vehicle analytics powered by Computer Vision.</p>', unsafe_allow_html=True)

# Main Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image Analysis", 
    "🎬 Video Upload", 
    "📹 Live Webcam",
    "🔗 Sample Demo"
])

# Shared Processing Function
def process_frame(frame, line_y, crossed_ids, total_crossed):
    results = model.track(frame, classes=[1, 2, 3, 5, 7], persist=True, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes
    curr_count = len(boxes)
    
    density = "LOW" if curr_count <= 2 else "MEDIUM" if curr_count <= 6 else "HIGH"
    density_color = "🟢" if density == "LOW" else "🟡" if density == "MEDIUM" else "🔴"

    if boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes.xyxy, ids):
            x1, y1, x2, y2 = map(int, box)
            cy = int((y1 + y2) / 2)
            if cy > line_y and track_id not in crossed_ids:
                crossed_ids.add(track_id)
                total_crossed += 1

    annotated_frame = results[0].plot()
    cv2.line(annotated_frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)
    
    return annotated_frame, curr_count, total_crossed, density, density_color

# --- Image Tab ---
with tab1:
    st.markdown("### Upload Image for Rapid Assessment")
    up_img = st.file_uploader("", type=["jpg", "png", "jpeg"], key="img_up")
    if up_img:
        img = Image.open(up_img)
        img_np = np.array(img)
        with st.spinner("Analyzing environment..."):
            results = model(img_np, conf=conf_threshold, classes=[1, 2, 3, 5, 7])
            res_img = results[0].plot()
            
            c1, c2 = st.columns(2)
            c1.image(img, caption="Original Stream", use_container_width=True)
            c2.image(res_img, caption="Deep Vision Result", use_container_width=True)
            
            st.success(f"Detections Finished: {len(results[0].boxes)} vehicles identified.")

# --- Video Tab ---
with tab2:
    st.markdown("### Detailed Video Analytics Corridor")
    up_vid = st.file_uploader("", type=["mp4", "avi", "mov"], key="vid_up")
    if up_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(up_vid.read())
        
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(3))
        height = int(cap.get(4))
        line_y = height // 2
        
        m1, m2, m3 = st.columns(3)
        count_box = m1.empty()
        crossed_box = m2.empty()
        density_box = m3.empty()
        
        st_frame = st.empty()
        stop = st.button("Terminate Process", key="stop_vid")
        
        crossed_ids = set()
        total_crossed = 0
        
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret: break
            
            p_frame, c_count, t_cross, dens, d_col = process_frame(frame, line_y, crossed_ids, total_crossed)
            total_crossed = t_cross
            
            count_box.metric("Current Traffic", c_count)
            crossed_box.metric("Throughput Volume", total_crossed)
            density_box.metric("Density Status", f"{d_col} {dens}")
            
            st_frame.image(p_frame, channels="BGR", use_container_width=True)
        
        cap.release()

# --- Webcam Tab ---
with tab3:
    st.markdown("### Real-Time Peripheral Monitoring")
    run_webcam = st.toggle("Activate Webcam")
    if run_webcam:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()
        
        crossed_ids = set()
        total_crossed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            line_y = frame.shape[0] // 2
            p_frame, c_count, t_cross, dens, d_col = process_frame(frame, line_y, crossed_ids, total_crossed)
            total_crossed = t_cross
            
            st_frame.image(p_frame, channels="BGR", use_container_width=True)
            
            if not run_webcam: break
        cap.release()

# --- Sample Tab ---
with tab4:
    st.markdown("### Instant Demonstration Capabilities")
    st.write("No video files handy? Try a professional dataset sample:")
    
    sample_options = {
        "Urban Intersection": "https://ultralytics.com/assets/anpr-demo-video.mp4",
        "Highway Flow": "https://ultralytics.com/assets/highway.mp4"
    }
    
    choice = st.selectbox("Select Sample Stream", list(sample_options.keys()))
    
    if st.button("Launch System Simulation"):
        st.info(f"Connecting to remote stream: {choice}...")
        cap = cv2.VideoCapture(sample_options[choice])
        st_frame = st.empty()
        
        # Metrics UI
        dm1, dm2 = st.columns(2)
        d_count = dm1.empty()
        d_cross = dm2.empty()
        
        crossed_ids = set()
        total_crossed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            line_y = frame.shape[0] // 2
            p_frame, c_count, t_cross, dens, d_col = process_frame(frame, line_y, crossed_ids, total_crossed)
            total_crossed = t_cross
            
            d_count.metric("Active Detection", c_count)
            d_cross.metric("Cumulative Count", total_crossed)
            
            st_frame.image(p_frame, channels="BGR", use_container_width=True)
            time.sleep(0.01) # Small delay for smooth playback
            
        cap.release()

# Footer
st.markdown("---")
st.markdown("© 2026 VisionFlow AI | Deep Learning Portfolio Project")
