# Attend-AI: Enterprise WebRTC Face Recognition System 🚀

Attend-AI is a high-performance, real-time facial recognition and attendance system. It leverages **NVIDIA GPU acceleration (TensorRT/CUDA)** and **WebRTC** to deliver a sub-100ms latency video stream with live AI annotations, achieving over **80 FPS** on local hardware.

---

## 💻 Hardware Requirements

To achieve the intended high-performance results (80+ FPS):
- **GPU**: NVIDIA RTX 30-series, 40-series, or 50-series (Tested on **RTX 5070**).
- **Environment**: Windows 10/11 or Ubuntu 20.04+.
- **Drivers**: NVIDIA Driver 525+, CUDA 11.8+, cuDNN 8.9+.

---

## 🛠️ Prerequisites

- **Python**: 3.10 or 3.11
- **Node.js**: 18.x or higher
- **MongoDB**: 6.0+ (Installed and running on `localhost:27017`)
- **Camera**: An RTSP-capable IP Camera.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```powershell
git clone https://github.com/coesecatria/deploy-version.git
cd deploy-version
```

### 2. Backend Setup
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model & AI Files Setup
The system requires specific pre-trained ONNX models for the RTX 5070 to accelerate.

**Required Models** (Must be in `backend/models/insightface/`):
- `det_10g.onnx`: SCRFD Detection model (Fastest/Accurate).
- `w600k_r50.onnx`: ArcFace Recognition model (Large-scale accuracy).

**Required Database Files** (Stored in project root):
- `student_index.faiss`: The vector database for face embeddings.
- `labels.pkl`: Mapping of embeddings to Roll Numbers.

You can download/ensure these are present by running:
```powershell
python download_models.py
```

### 4. Frontend Setup
```powershell
cd ../frontend
npm install
```

---

## ⚙️ Configuration

Open `backend/app/core/config.py` and set your RTSP URL:

```python
class Settings:
    ip_camera_url = "rtsp://admin:YourPassword@172.16.x.x:554/stream"
    mongodb_url = "mongodb://localhost:27017"
```

---

## 🏃 Running the Application

### Start the Backend
In the `backend` folder (with `venv` active):
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Start the Frontend
In the `frontend` folder:
```powershell
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## 🧬 AI Database Management

### Initial Re-indexing
If you have images in `processed_dataset/`, run the bulk re-indexer to align and vectorize them on your GPU:
```powershell
python reindex_all.py
```

### Adding New Students
1. Log in as Admin (**PIN: `Attendence_cybersec`**).
2. Go to the **Register** page.
3. The system will use your **Local Webcam** to capture 15 high-quality frames.
4. The FAISS index will update in real-time.

---

## 💎 Key Features
- **WebRTC Streaming**: Hardware-accelerated video delivery with zero browser lag.
- **720p HD AI**: Crystal-clear detection at 1280x720 resolution.
- **ByteTrack**: Advanced multi-object tracking for crowded environments (up to 70 faces).
- **GPU Optimization**: Full TensorRT integration for sub-millisecond recognition.
- **Unknown Detection**: Red-box labeling for high-security awareness.

---

## 🛠️ Troubleshooting
- **Black Screen on Kiosk**: Ensure the RTSP URL is correct and the backend logs show "Connecting to RTSP...".
- **CUDA Errors**: Verify `onnxruntime-gpu` is installed and CUDA DLLs are in your system PATH.
- **Proxy Errors**: Ensure the backend is running on `port 8000` before starting the frontend.

---
*Built for High-Performance AI Surveillance.*
