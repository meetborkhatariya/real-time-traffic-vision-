# VisionFlow AI: Microservice Traffic Intelligence 🚦🤖

A production-ready computer vision architecture designed for real-time traffic management. This project moves beyond simple scripting by implementing a **Microservice Architecture** separating the deep learning engine (FastAPI) from the user interface (Streamlit) while persisting analytics data (SQLite/SQLAlchemy).

## 🌟 Architectural Features
- **FastAPI Backend Core**: A RESTful and WebSocket API layer that wraps the YOLOv8 neural network.
- **Decoupled Streamlit Dashboard**: A lightweight frontend that authenticates and consumes the API without doing any heavy deep learning locally.
- **SQL Data Pipeline**: Uses SQLAlchemy ORM to log every single vehicle crossing, timestamp, and class into an SQLite database for business intelligence.
- **Dynamic Hot-Swapping**: Ability to hot-swap YOLO weights (Nano vs Large) via an API `POST` request without restarting the server.
- **Multi-Source Streaming**: Processes static images, raw video files, live webcam hardware, and cloud URLs natively.

## 🏗️ Project Flow & Engineering Logic
This system is engineered to solve real-world MLOps problems:

1. **The Vector State-Machine (Line Crossing):** Unlike basic threshold trackers that count false-positives when cars park on a line, this architecture uses a mathematical Vector State-Machine. The backend stores the `(x, y)` pixels of every tracked ID in Python memory. It only writes to the database if a vehicle actively moves from *above* the Y-line to *below* the Y-line between frames.
2. **Local Hardware Access (Webcam Edge Computing):** When you click "Start API Camera" in the dashboard, the frontend routes an HTTP request to the backend microservice. The FastAPI server physically opens hardware access to your laptop's camera (which triggers the physical camera light to turn on), streams the raw matrices through YOLO, and yields a processed MJPEG bytes stream back perfectly over the HTTP protocol. Disconnecting from the Streamlit UI instantly kills the Python generator loop, releasing the hardware and turning the light off safely.
3. **Ghost Filtering & Permanence:** By tweaking ByteTrack hyperparameters, the AI drops "ghost" background objects if they fail to persist, and remembers real cars for up to 5 seconds if they are temporarily occluded by trees or buses.

## 📊 Business Value Dashboard
By persisting tracking metadata to an SQLite database (`traffic_analytics.db`), the frontend can pull live analytics.
- **Traffic Optimization**: Tracks throughput volumes.
- **Vehicle Classification**: Distinguishes between Cars, Trucks, Motorcycles, and Buses.
- **State Preservation**: The backend maintains event logs even if the frontend crashes or goes offline.

## 🚀 Installation & Setup

This project requires spinning up two microservices.

### 1. Start the API Engine (Backend)
Navigate to the `backend` directory and launch the FastAPI server.
```bash
# Install dependencies
pip install fastapi uvicorn sqlalchemy opencv-python-headless ultralytics numpy

# Start the server on port 8000
cd backend
python -m uvicorn main:app --port 8000 --reload
```

### 2. Start the UI Dashboard (Frontend)
Open a new terminal window / tab, stay in the root directory, and launch the Streamlit frontend.
```bash
# Install dependencies
pip install streamlit requests pandas

# Launch the UI
streamlit run app.py
```

## 🧠 Model Intelligence & Accuracy
This project provides 4 different levels of AI intelligence to balance speed (FPS) and accuracy (mAP).

| Mode | Model | mAP@50-95 | Latency (CPU) | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Fast** | YOLOv8n | 37.3 | ~1ms | Mobile devices, edge computing (Raspberry Pi). |
| **Balanced** | YOLOv8s | 44.9 | ~2ms | General real-time tracking on laptops. |
| **Precise** | YOLOv8m | 50.2 | ~4ms | Static images, night-time traffic. |
| **Ultra Precise**| YOLOv8l | 52.9 | ~8ms | Cloud servers, toll-collection logic on remote APIs. |

## 🛠️ Performance Tech Stack
- **AI Core**: YOLOv8 (Ultralytics), OpenCV
- **Backend**: Python 3.8+, FastAPI, Uvicorn (ASGI)
- **Database**: SQLite, SQLAlchemy ORM
- **Frontend**: Streamlit, Requests, Pandas
- **Architecture**: Microservices, REST API, MJPEG Streaming

---
*Developed as a Placement-Ready AI Engineer Portfolio Project.*
