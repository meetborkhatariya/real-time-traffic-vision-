import streamlit as st
import requests
import pandas as pd
import base64
from PIL import Image
import io

# Set Page Config
st.set_page_config(
    page_title="VisionFlow AI | Microservices",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(circle at top left, #1a1c2c, #0e1117); }
    .main-header {
        background: linear-gradient(90deg, #00d4ff, #0080ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3.5rem; margin-bottom: 0px;
    }
    .stButton>button {
        width: 100%; border-radius: 10px; background: linear-gradient(90deg, #00d4ff, #0080ff);
        color: white; font-weight: bold; border: none; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4); }
</style>
""", unsafe_allow_html=True)

API_URL = "https://traffic-vision-backend.onrender.com"

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965214.png", width=100)
    st.title("VisionFlow AI")
    st.markdown("---")
    
    st.subheader("⚙️ Backend Intelligence")
    api_online = False
    with st.spinner("Connecting to API engine... (Render cold starts take ~45–90s)"):
        try:
            # Increase timeout to 120s for Render free tier
            res = requests.get(f"{API_URL}/ping", timeout=120)
            if res.status_code == 404:
                res = requests.get(f"{API_URL}/", timeout=120)

            if res.status_code == 200:
                st.success("API Engine: ONLINE 🟢")
                api_online = True
            else:
                st.error(f"API Engine: ERROR 🔴 (Status {res.status_code})")
        except requests.Timeout:
            st.warning("⏳ API is waking up. Render's free tier spins down after 15m. Please refresh in 30s.")
        except Exception as e:
            st.error("API Engine: OFFLINE 🔴")
            st.info("Ensure the Render backend is deployed and not crashing due to RAM limits.")
        
    st.markdown("---")
    
    st.subheader("🧠 Model Version")
    st.caption("⚠️ Render Free Tier has 512MB RAM. Avoid 'Ultra Precise' if the API crashes.")
    model_map = {
        "Fast (YOLOv8n)": "yolov8n.pt",
        "Balanced (YOLOv8s)": "yolov8s.pt",
        "Ultra Precise (YOLOv8l)": "yolov8l.pt"
    }
    
    model_choice = st.selectbox("Select Cloud Model", list(model_map.keys()))
    if st.button("Switch Backend Model"):
        if api_online:
            with st.spinner("Instructing backend to switch weights..."):
                r = requests.post(f"{API_URL}/api/config/model", json={"model_name": model_map[model_choice]}, timeout=120)
                if r.status_code == 200:
                    st.success("Model updated successfully!")
        else:
            st.error("API is offline.")
            
    conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.35)
            
    st.markdown("---")
    st.subheader("📊 Analytics Dashboard")
    if st.button("🔄 Refresh Data"):
        st.rerun()

    if api_online:
        try:
            stats = requests.get(f"{API_URL}/api/analytics/summary", timeout=10).json()
            total = stats.get("total_crossed", 0)
            breakdown = stats.get("breakdown", {})
            st.metric("Total Vehicles Handled via API", total)
            if breakdown:
                st.write("**Vehicle Breakdown:**")
                df = pd.DataFrame(list(breakdown.items()), columns=["Type", "Count"])
                st.bar_chart(df.set_index("Type"))
                
            st.markdown("---")
            st.write("**Raw Database Logs (Last 50):**")
            raw_data = requests.get(f"{API_URL}/api/analytics/data", timeout=10).json()
            if raw_data:
                raw_df = pd.DataFrame(raw_data)
                # Format time for readability
                raw_df['Time'] = pd.to_datetime(raw_df['Time']).dt.strftime('%H:%M:%S')
                st.dataframe(raw_df, hide_index=True)
                
        except Exception as e:
            st.info(f"Could not load analytics. {e}")

st.markdown('<p class="main-header">VisionFlow Database & UI</p>', unsafe_allow_html=True)
st.markdown('<p style="color: #8b949e; font-size: 1.2rem; margin-top: -10px;">Fully Restored Microservices UI</p>', unsafe_allow_html=True)

# Restored all 4 tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image API", 
    "🎬 Video Upload", 
    "📹 Live Webcam API",
    "🔗 Flow Demo"
])

def play_mjpeg_stream(url, method="post", kwargs=None):
    st_frame = st.empty()
    stop_btn = st.button("🛑 Stop Stream")
    with st.spinner("Connecting to backend... (Render may need ~30–60s to wake up)"):
        try:
            kwargs = kwargs or {}
            kwargs["stream"] = True
            kwargs["timeout"] = 90  # Allow time for Render cold start
            response = requests.request(method, url, **kwargs)
        except requests.Timeout:
            st.warning("⏳ The backend timed out while waking up. Please wait 30 seconds and try again.")
            return
        except requests.ConnectionError:
            st.error("❌ Cannot reach the backend. Check your internet or Render deployment status.")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return

    bytes_data = bytes()
    try:
        for chunk in response.iter_content(chunk_size=16384):
            if stop_btn:
                break
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                st_frame.image(jpg, use_column_width=True)
    except Exception as e:
        st.error(f"❌ Stream interrupted: {e}")

# Restored Image Tab via API
with tab1:
    st.markdown("### Process Images via API Endpoint")
    up_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="img_up", label_visibility="collapsed")
    if up_img:
        st.image(up_img, caption="Frontend Original", width=300)
        if st.button("Send to Deep Learning API"):
            with st.spinner("Processing in cloud..."):
                # Resize locally before upload to save bandwidth and reduce latency
                img = Image.open(up_img)
                max_size = 1280
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Convert back to bytes
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                img_bytes = buf.getvalue()

                file_obj = {"file": (up_img.name, img_bytes, "image/jpeg")}
                try:
                    r = requests.post(f"{API_URL}/api/image/process?conf={conf_threshold}", files=file_obj, timeout=120)
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"Detections Finished: {data['count']} vehicles.")
                        img_bytes_res = base64.b64decode(data['image'])
                        st.image(Image.open(io.BytesIO(img_bytes_res)), caption="API Processed Result", use_column_width=True)
                    else:
                        st.error(f"API Error ({r.status_code}): {r.text}")
                except requests.exceptions.ReadTimeout:
                    st.error("⏰ The API took too long to respond (Timeout). This usually happens if the backend is waking up or processing a complex image. Please try again in a moment.")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {e}")

# Video Upload
with tab2:
    st.markdown("### Process Video via API")
    up_vid = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="vid_up", label_visibility="collapsed")
    if up_vid:
        if st.button("🚀 Analyze Video"):
            file_obj = {"file": (up_vid.name, up_vid.getvalue(), up_vid.type)}
            play_mjpeg_stream(f"{API_URL}/api/video/stream?conf={conf_threshold}", method="post", kwargs={"files": file_obj})

# Restored Webcam Tab via API
with tab3:
    st.markdown("### Connect to Local Edge Camera via API")
    st.warning("This instructs the backend to open its local camera (webcam 0).")
    if st.button("🎥 Start API Camera"):
        play_mjpeg_stream(f"{API_URL}/api/video/webcam?conf={conf_threshold}", method="get")

# URL Demo
with tab4:
    st.markdown("### Process Cloud URLs")
    sample_options = {
        "Urban Intersection": "https://ultralytics.com/assets/anpr-demo-video.mp4"
    }
    choice = st.selectbox("Select Sample", list(sample_options.keys()))
    if st.button("🔥 Launch API Simulation"):
        url_payload = {"url": sample_options[choice], "conf": conf_threshold}
        play_mjpeg_stream(f"{API_URL}/api/video/stream_url", method="post", kwargs={"json": url_payload})
