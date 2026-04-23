# VisionFlow AI: Real-Time Traffic Intelligence 🚦🤖

A cutting-edge computer vision dashboard designed for modern traffic management. Powered by **YOLOv8** and **Streamlit**, this system provides deep insights into vehicle movement, density, and flow volume.

## 🌟 Key Features
- **Multi-Source Input**: Analyze traffic via **Image Uploads**, **Video Files**, or **Live Webcam** feeds.
- **Deep Tracking**: Leverages ByteTrack/BoT-SORT for persistent vehicle identification.
- **Smart Counting**: Precision line-crossing algorithm to monitor traffic throughput.
- **Density Analytics**: Real-time traffic state classification (Low, Medium, High).
- **Interactive UI**: Custom-styled dashboard with live metrics and terminology.
- **Instant Samples**: Built-in sample streams for quick demonstrations.

## 📊 Model Intelligence & Accuracy
This project provides 4 different levels of AI intelligence to balance speed (FPS) and accuracy (mAP).

| Mode | Model | mAP@50-95 | Latency (CPU) | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Fast** | YOLOv8n | 37.3 | ~1ms | Mobile devices, high-speed traffic video. |
| **Balanced** | YOLOv8s | 44.9 | ~2ms | General real-time tracking on laptops. |
| **Precise** | YOLOv8m | 50.2 | ~4ms | Static images, detailed density analysis. |
| **Ultra Precise**| YOLOv8l | 52.9 | ~8ms | Server-side auditing, toll-collection logic. |

> **Note**: For placements/demos, **Balanced (v8s)** is recommended as the default.

## 🛠️ Performance Tech Stack
- **AI Core**: YOLOv8 (Ultralytics)
- **Deployment**: Streamlit Cloud
- **Logic**: Python 3.8+, OpenCV, NumPy
- **Visuals**: custom HSL Gradients & Glassmorphism UI

## 🚀 Installation & Setup

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/visionflow-ai.git
   cd visionflow-ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## 📊 Analytics Workflow
1. **Detection**: Identifying vehicles (Cars, Trucks, Buses, Cycles).
2. **Persistence**: Tracking unique IDs across frames.
3. **Trigger**: Counting when a vehicle's centroid crosses the red threshold line.
4. **Output**: Live visualization of bounding boxes, track trails, and metrics.

---
*Created for an AI/ML Computer Vision Portfolio.*
