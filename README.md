<div align="center">

# 🎓 Attend-AI
### Real-Time Face Recognition Attendance System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.131-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

> A high-performance, GPU-accelerated facial recognition attendance system with live WebRTC video streaming, FAISS vector search, and a beautiful React dashboard.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [AI Database Management](#-ai-database-management)
- [Application Pages](#-application-pages)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

**Attend-AI** is an enterprise-grade, real-time attendance management system that uses cutting-edge facial recognition to automatically mark student attendance via an IP camera or webcam. It combines **InsightFace ONNX models** on GPU, a **FAISS vector database** for sub-millisecond face lookup, and a **React + Vite** dashboard for live monitoring, reporting, and student management.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **Live WebRTC Stream** | Hardware-accelerated video from IP/RTSP camera with AI bounding boxes |
| 🧠 **GPU Face Recognition** | InsightFace ArcFace + SCRFD via ONNX Runtime on CUDA |
| ⚡ **80+ FPS Processing** | Real-time inference on RTX 30/40/50-series GPUs |
| 🗃️ **FAISS Vector Search** | Lightning-fast identity lookup across thousands of students |
| 📊 **Analytics Dashboard** | Charts, trends, and branch-wise attendance breakdowns |
| 📄 **PDF Export** | Branded, color-coded attendance reports with matplotlib charts |
| 🔐 **Admin PIN Auth** | Secure admin login with role-based page access |
| 🏫 **Multi-Branch Support** | CSE, ECE, EEE, MECH, CIVIL, MBA, MCA and more |
| 📅 **Date-Range Filtering** | Attendance queries by date, branch, or individual student |
| 📷 **Webcam Registration** | Register new students directly via browser webcam (15 frames) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)              │
│  Dashboard │ Kiosk │ Students │ Attendance │ Analytics │ PDF│
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP / WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + Uvicorn)               │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Face Engine  │  │ Stream Mgr   │  │ Report Generator │  │
│  │ InsightFace  │  │ RTSP / MJPEG │  │ Jinja2 + MatPlot │  │
│  │ FAISS Lookup │  │ OpenCV + GPU │  │ PDF via WeasyPrint│  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌────────┐     ┌──────────────┐
│MongoDB │     │  FAISS Index │
│Attend. │     │  labels.pkl  │
│Records │     │  (root dir)  │
└────────┘     └──────────────┘
```

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA GTX 1660 (CUDA 11.8) | NVIDIA RTX 3070+ / 5070 |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 5 GB free | 10 GB+ |
| **OS** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| **NVIDIA Driver** | 525+ | Latest |

> ⚠️ **CPU-only mode is not recommended** — face recognition inference will be extremely slow without a CUDA-capable GPU.

---

## 🛠️ Prerequisites

Make sure the following are installed **before** proceeding:

### 1. Python 3.10 or 3.11
```powershell
# Verify version
python --version   # Should output Python 3.10.x or 3.11.x
```
Download: https://www.python.org/downloads/

### 2. Node.js 18.x or higher
```powershell
node --version   # Should output v18.x.x or higher
npm --version
```
Download: https://nodejs.org/en/download

### 3. MongoDB 6.0+
```powershell
# Verify MongoDB is running
mongosh --eval "db.runCommand({ connectionStatus: 1 })"
```
Download: https://www.mongodb.com/try/download/community  
> Make sure MongoDB service is **started** and listening on `localhost:27017`.

### 4. NVIDIA CUDA Toolkit 11.8+
```powershell
nvcc --version    # Should show CUDA 11.8 or higher
nvidia-smi        # Verify GPU is detected
```
Download: https://developer.nvidia.com/cuda-toolkit

### 5. Git
```powershell
git --version
```
Download: https://git-scm.com/downloads

---

## 📦 Installation

### Step 1 — Clone the Repository

```powershell
git clone https://github.com/coesecatria/deploy-version.git
cd deploy-version
```

---

### Step 2 — Backend Setup

```powershell
# Navigate to the project root (where requirements.txt lives)
cd deploy-version

# Create a Python virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Install PyTorch with CUDA support **first** (before requirements.txt)

> ⚠️ This step is critical. Installing PyTorch from pip without CUDA support will break GPU acceleration.

```powershell
# For CUDA 12.8 (RTX 40/50-series) — check https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# For CUDA 11.8 (RTX 30-series / older)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is detected
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

#### Install remaining Python dependencies

```powershell
pip install -r requirements.txt
```

---

### Step 3 — Download AI Models

The system requires InsightFace ONNX models. Run the provided script from **inside the `backend/` folder**:

```powershell
cd backend
python download_models.py
```

This will download and extract `buffalo_l.zip` (~275 MB) into `backend/models/insightface/`.

After extraction, verify the following files exist:

```
backend/models/insightface/
├── det_10g.onnx       ← SCRFD face detector
├── w600k_r50.onnx     ← ArcFace recognition model
├── 1k3d68.onnx        ← 3D landmark model
├── 2d106det.onnx      ← 2D landmark model
└── genderage.onnx     ← Gender & age estimation
```

---

### Step 4 — Configure Environment Variables

```powershell
# From the project root
cp .env.example .env
```

Open `.env` and fill in your values:

```env
# MongoDB
MONGO_URI=mongodb://localhost:27017
DB_NAME=attendance_db

# Shift timings (24-hour format)
LOGIN_TIME=09:30:00
LOGOUT_TIME=16:30:00

# FAISS face match threshold (0.0 to 1.0, higher = stricter)
SIMILARITY_THRESHOLD=0.70

# Frontend API base URL
VITE_API_URL=http://localhost:8000/api
```

---

### Step 5 — Configure RTSP Camera

Open `backend/app/core/config.py` and update your camera URL:

```python
class Settings:
    # RTSP URL format for IP cameras (CP-Plus, Hikvision, Dahua, etc.)
    ip_camera_url: str = "rtsp://admin:YourPassword@192.168.1.64:554/stream1"
    
    # MongoDB connection
    mongodb_url: str = "mongodb://localhost:27017"
```

> 💡 **No IP Camera?** The system can fall back to a local webcam. Set `ip_camera_url = 0` to use the default webcam (index 0).

---

### Step 6 — Frontend Setup

```powershell
# Navigate to frontend folder
cd ../frontend

# Install Node.js dependencies
npm install
```

---

## ▶️ Running the Application

Open **two separate terminal windows** (both with the venv activated):

### Terminal 1 — Start the Backend

```powershell
# From the project root, with venv active
.\venv\Scripts\Activate.ps1

# Start FastAPI backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir backend
```

You should see:
```
============================================================
  ATTEND-AI — Starting up...
============================================================
  [Face Engine] Loading InsightFace models on CUDA...
  [Database] Connected to MongoDB
  [Streamer] Connecting to RTSP camera...
============================================================
  Server ready! Docs at http://localhost:8000/docs
============================================================
```

### Terminal 2 — Start the Frontend

```powershell
# From the frontend/ folder
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in 300 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

### ✅ Open the App

| URL | Description |
|---|---|
| `http://localhost:5173` | Main React Application |
| `http://localhost:8000/docs` | FastAPI Swagger UI (API Explorer) |
| `http://localhost:8000` | Health Check Endpoint |

---

## 🧬 AI Database Management

### Building the FAISS Index from Scratch

If you have student images in `processed_dataset/`, run the bulk re-indexer from the **project root**:

```powershell
# With venv active, from project root
python reindex_all.py
```

This will:
1. Scan all images in `processed_dataset/`
2. Generate 512-dimensional face embeddings via ArcFace on GPU
3. Build and save `student_index.faiss` and `labels.pkl` to the project root

### Registering a New Student via Web UI

1. Navigate to `http://localhost:5173`
2. Log in with the Admin PIN: **`Attendence_cybersec`**
3. Go to the **Register** page
4. Fill in the student's details (Name, Roll No., Branch, Year)
5. Click **Start Camera** — the system captures **15 high-quality frames** using your webcam
6. Click **Register** — the FAISS index updates in real-time

> 💡 Ensure good lighting and have the student face the camera directly for best recognition accuracy.

---

## 📱 Application Pages

| Page | URL | Description |
|---|---|---|
| **Login** | `/login` | Admin PIN authentication |
| **Dashboard** | `/dashboard` | Live stats: total students, today's attendance, on-time vs late |
| **Kiosk** | `/kiosk` | Full-screen live camera feed with real-time face recognition overlay |
| **Students** | `/students` | Browse, search, and manage all registered students |
| **Attendance** | `/attendance` | View and filter attendance logs by date, branch, or student |
| **Analytics** | `/analytics` | Charts for trends, branch-wise breakdown, and weekly reports |
| **Register** | `/register` | Register new students with live webcam capture |
| **Camera** | `/camera` | Camera stream diagnostics and configuration |

---

## 🔌 API Reference

The backend exposes a RESTful API at `http://localhost:8000/api`. Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/students` | List all registered students |
| `POST` | `/api/students/register` | Register a new student with face embeddings |
| `GET` | `/api/attendance` | Get attendance records (supports date/branch filters) |
| `GET` | `/api/attendance/today` | Get today's attendance summary |
| `GET` | `/api/reports/export` | Export attendance as a PDF report |
| `GET` | `/api/stream/feed` | MJPEG live video stream with AI overlay |
| `GET` | `/api/settings` | Get/update global system settings |
| `GET` | `/` | Health check — returns GPU & engine status |

> Full interactive docs: **`http://localhost:8000/docs`**

---

## 📁 Project Structure

```
deploy-version/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/          # FastAPI route handlers
│   │   │       ├── students.py
│   │   │       ├── attendance.py
│   │   │       ├── streaming.py
│   │   │       ├── reports.py
│   │   │       ├── settings.py
│   │   │       └── webrtc.py
│   │   ├── core/
│   │   │   ├── config.py        # App settings & RTSP URL
│   │   │   └── database.py      # MongoDB connection
│   │   ├── models/              # Pydantic data models
│   │   ├── services/
│   │   │   ├── face_engine.py   # InsightFace + FAISS engine
│   │   │   └── stream_manager.py # RTSP capture & AI pipeline
│   │   ├── templates/
│   │   │   └── attendance_report.html  # Jinja2 PDF template
│   │   ├── main.py              # FastAPI app + lifespan
│   │   └── seed.py              # DB seeder from metadata.csv
│   ├── models/
│   │   └── insightface/         # ONNX model files (not in git)
│   └── download_models.py       # Model downloader script
│
├── frontend/
│   ├── src/
│   │   ├── pages/               # React page components
│   │   ├── components/          # Shared UI components
│   │   └── main.jsx             # App entry point
│   ├── package.json
│   └── vite.config.js
│
├── processed_dataset/           # Student face images (not in git)
├── student_index.faiss          # FAISS vector index (not in git)
├── labels.pkl                   # Embedding labels (not in git)
├── reindex_all.py               # Bulk FAISS re-indexer
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
└── .gitignore
```

---

## 🔧 Troubleshooting

### ❌ Black Screen on Kiosk Page
- Verify the RTSP URL is correct in `config.py`
- Check backend logs for `"Connecting to RTSP..."` or connection errors
- Test the URL in VLC: `Media → Open Network Stream → paste RTSP URL`
- Try setting `ip_camera_url = 0` for local webcam fallback

### ❌ CUDA / GPU Not Detected
```powershell
# Check if ONNX Runtime sees the GPU
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include 'CUDAExecutionProvider'
```
- Ensure `onnxruntime-gpu` is installed (not `onnxruntime`)
- Verify CUDA DLLs are in your system `PATH`
- Re-install PyTorch with the correct CUDA version

### ❌ MongoDB Connection Error
```powershell
# Start MongoDB service (Windows)
net start MongoDB

# Or manually
mongod --dbpath "C:\data\db"
```
- Ensure MongoDB is running on `localhost:27017`
- Check `MONGO_URI` in your `.env` file

### ❌ Frontend Proxy / CORS Errors
- Make sure the backend is fully started on **port 8000** before launching the frontend
- Check `VITE_API_URL=http://localhost:8000/api` is set in `.env`

### ❌ `pip install` Fails for InsightFace / ONNX
```powershell
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Then retry
pip install insightface onnxruntime-gpu
```

### ❌ Face Not Recognized (Unknown Box)
- Check `SIMILARITY_THRESHOLD` in `.env` — lower it slightly (e.g., `0.60`)
- Re-register the student in better lighting conditions
- Run `python reindex_all.py` to rebuild the FAISS index from scratch

---

## 📄 License

This project is built for educational and institutional use by the **COE Security & AI Research Group**.

---

<div align="center">

**Built with ❤️ for smarter campuses.**  
*Powered by InsightFace · FastAPI · React · MongoDB · FAISS · CUDA*

</div>
