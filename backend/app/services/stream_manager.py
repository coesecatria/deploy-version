import cv2
import time
import asyncio
import numpy as np
import threading
import traceback
from typing import Optional, List, Dict

from app.services.face_engine import engine
from app.services.scrfd import SCRFD
from app.utils.alignment import align_face
from app.services.recognizer import _get_detector
from app.core.config import settings
import os

# FFmpeg/OpenCV Stability: Use 2 threads for general work
cv2.setNumThreads(2)
# Global environment variable for FFmpeg options
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|threads;1"
from ultralytics.trackers.byte_tracker import BYTETracker

class EnterpriseStreamManager:
    def __init__(self):
        # ── Capture Loop (Producer) ──
        self.rtsp_url = settings.ip_camera_url
        self.cap = None
        self._capture_thread = None
        self._stop_event = threading.Event()
        
        # ── Shared State (Thread-Safe) ──
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        
        # ── Process Loop (Consumer) ──
        self._process_thread = None
        
        # ── MJPEG Loop (Broadcaster) ──
        self._mjpeg_thread = None
        self._new_frame_event = threading.Event()
        
        # ── Annotations (Thread-Safe) ──
        self._current_detections = []
        self._det_lock = threading.Lock()
        
        # ── ByteTracker State ──
        self.tracker = None 
        
        # ── Identity Cache ──
        self.identity_cache = {} 
        self.last_scan_time = {} # For DB cooldowns
        
        # ── Global AI-Annotated Buffer (Optimized for WebRTC) ──
        self._latest_annotated_frame = None
        self._annotated_lock = threading.Lock()
        self._latest_jpeg = None
        self._jpeg_lock = threading.Lock()
        self.enable_mjpeg = True # Flag to avoid redundant encoding load
        
        # ── Lifecycle Locks ──
        self._start_lock = threading.Lock()
        self._cap_lock = threading.Lock() # 🔒 Ensure read/release NEVER happen at once
        
        # ── Async Synchronization (Multi-subscriber support) ──
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: List[asyncio.Queue] = []
        self._sub_lock = threading.Lock()

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Starts the decoupled Producer (Camera) and Consumer (AI) threads safely."""
        with self._start_lock:
            # 1. If already active, don't restart (idempotent)
            if not self._stop_event.is_set() and self._capture_thread and self._capture_thread.is_alive():
                print("  [StreamManager] Stream already running. Skipping start.")
                return

            print("  [StreamManager] Initializing backend threads...")
            
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
            self.main_loop = loop
            self._stop_event.clear()
            
            # Start Threads
            self._capture_thread = threading.Thread(target=self._capture_loop, name="ProducerThread", daemon=True)
            self._process_thread = threading.Thread(target=self._process_loop, name="ConsumerThread", daemon=True)
            self._mjpeg_thread = threading.Thread(target=self._mjpeg_loop, name="BroadcasterThread", daemon=True)
            
            self._capture_thread.start()
            self._process_thread.start()
            self._mjpeg_thread.start()
            print("  [StreamManager] All backend threads started.")
        
    def stop(self):
        """Safely signals all threads to stop and releases hardware resources."""
        self._stop_event.set()
        with self._cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        
        # Ensure latest frame is cleared so WebRTC doesn't stream stale data
        with self._frame_lock:
            self._latest_frame = None
        with self._annotated_lock:
            self._latest_annotated_frame = None
            
    def pause(self):
        print("  [StreamManager] Pausing stream...")
        self._stop_event.set()
        with self._cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            
    def resume(self):
        print("  [StreamManager] Resuming stream...")
        if self.main_loop:
            self.start(self.main_loop)
            
    # ── Producer Thread (Zero-Latency Video) ──────────────────────────────────
    
    def _capture_loop(self):
        """Dedicated thread to pull frames from RTSP without delay."""
        print(f"  [StreamManager] Starting Enterprise Capture: {self.rtsp_url}")
        
        try:
            # Ensure FFmpeg options are set for TCP and Single-Thread stability
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|threads;1"
            
            while not self._stop_event.is_set():
                try:
                    with self._cap_lock:
                        if self.cap: self.cap.release()
                        print(f"  [StreamManager] Connecting to RTSP... {self.rtsp_url}")
                        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                        
                        if not self.cap.isOpened():
                            print("  [StreamManager] Link failed. Retrying in 5s...")
                        else:
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if self.cap is None or not self.cap.isOpened():
                        time.sleep(5)
                        continue
                    
                    # Target resolution for Kiosk (1280x720 720p HD)
                    self.TARGET_WIDTH = 1280 
                    self.TARGET_HEIGHT = 720 
                    
                    self._last_mjpeg_time = 0
                    self._mjpeg_interval = 1.0 / 20.0 # 20 FPS balance

                    frame_count = 0
                    start_time = time.time()
                    self._source_h, self._source_w = 0, 0 

                    while not self._stop_event.is_set():
                        ret = False
                        frame = None
                        try:
                            with self._cap_lock:
                                if self.cap:
                                    ret, frame = self.cap.read()
                        except Exception as read_err:
                            print(f"  [StreamManager] Low-level read error: {read_err}")
                            break 
                        
                        # Set source resolution once
                        if self._source_h == 0:
                            self._source_h, self._source_w = frame.shape[:2]
                            
                        # Store latest frame
                        with self._frame_lock:
                            self._latest_frame = frame.copy()
                        
                        # Signal MJPEG and AI threads
                        self._new_frame_event.set()
                        
                        frame_count += 1
                        if frame_count % 300 == 0:
                            fps = frame_count / (time.time() - start_time)
                            print(f"  [StreamManager] Capture FPS: {fps:.2f}")

                        time.sleep(0.001) 
                    
                except Exception as inner_e:
                    print(f"  [StreamManager] Loop error: {inner_e}")
                    time.sleep(2)
                finally:
                    if self.cap: self.cap.release()
                    
        except Exception as e:
            print(f"  [StreamManager] FATAL Global Capture Error: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"  [StreamManager] FATAL Capture Error: {e}")
            traceback.print_exc()

    # ── Consumer Thread (Heavy AI Pipeline) ──────────────────────────────────
    
    def _init_tracker(self):
        from types import SimpleNamespace
        args = SimpleNamespace(
            tracker_type='bytetrack', 
            track_high_thresh=0.5, 
            track_low_thresh=0.1, 
            new_track_thresh=0.6, 
            track_buffer=30, 
            match_thresh=0.8,
            fuse_score=True
        )
        self.tracker = BYTETracker(args=args, frame_rate=30)
        
    def _format_for_bytetrack(self, bboxes, scores):
        if len(bboxes) == 0:
            return np.empty((0, 6))
        tracks = np.zeros((len(bboxes), 6), dtype=np.float32)
        tracks[:, :4] = bboxes
        tracks[:, 4] = scores
        tracks[:, 5] = 0 # Class 0
        return tracks

    def _process_loop(self):
        """Async processing thread for AI logic."""
        print("  [StreamManager] Starting AI Consumer Thread")
        
        try:
            engine.initialize()
            
            # 🔥 FAISS GPU Optimization: Move index to GPU for sub-millisecond search
            try:
                import faiss
                res = faiss.StandardGpuResources()
                engine.index = faiss.index_cpu_to_gpu(res, 0, engine.index)
                print("  [StreamManager] FAISS moved to GPU 🔥")
            except Exception as fe:
                print(f"  [StreamManager] FAISS GPU fallback: {fe}")

            detector = _get_detector()
            self._init_tracker()
            
            proc_count = 0
            start_time = time.time()

            while not self._stop_event.is_set():
                frame = None
                with self._frame_lock:
                    if self._latest_frame is not None:
                        frame = self._latest_frame.copy()
                
                if frame is None:
                    # Minor wait to avoid spinning
                    time.sleep(0.005)
                    continue
                    
                # 1. Detection
                bboxes, scores, kpss = detector.detect(frame, thresh=0.5)
                
                if bboxes is None or len(bboxes) == 0:
                    with self._det_lock:
                        self._current_detections = []
                    continue
                    
                # 2. Tracking (ByteTrack)
                import torch
                from ultralytics.engine.results import Boxes
                try:
                    track_input = self._format_for_bytetrack(bboxes, scores)
                    det = Boxes(torch.tensor(track_input), frame.shape[:2])
                    tracks = self.tracker.update(det, frame)
                except Exception as track_err:
                    print(f"  [StreamManager] Tracker warning: {track_err}")
                    with self._det_lock:
                        self._current_detections = [{"box": b.astype(int), "text": "Detecting...", "sim": 0, "color": (0,0,255)} for b in bboxes]
                    continue
                
                new_detections = []
                now_ts = time.time()
                
                # tracks: [x1, y1, x2, y2, id, conf, cls, idx]
                for t in tracks:
                    x1, y1, x2, y2, t_id, conf, cls, idx = t
                    t_id = int(t_id)
                    box_center = np.array([(x1+x2)/2, (y1+y2)/2])
                    
                    centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in bboxes])
                    dists = np.linalg.norm(centers - box_center, axis=1)
                    best_idx = np.argmin(dists)
                    landmarks = kpss[best_idx]
                    
                    identity = self.identity_cache.get(t_id)
                    needs_embedding = (identity is None) or (now_ts - identity["last_updated"] > 2.0)
                        
                    roll_no = "Unknown"
                    color = (0, 0, 255)
                    sim = 0.0
                    
                    if needs_embedding:
                        face_aligned = align_face(frame, landmarks)
                        if face_aligned is not None:
                            embedding = engine.get_embedding(cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB))
                            embedding = np.expand_dims(embedding, axis=0).astype(np.float32)
                            distances, indices = engine.index.search(embedding, k=1)
                            
                            found_sim = float(distances[0][0])
                            found_idx = int(indices[0][0])
                            
                            # HIGHER is BETTER for IndexFlatIP (Inner Product / Cosine)
                            # Threshold 0.65 is standard for high-security ArcFace
                            if found_idx >= 0 and found_sim > 0.65:
                                roll_no = engine.labels[found_idx]
                                sim = found_sim # It's already Cosine
                                color = (0, 255, 0) # Green for Recognized
                                self.identity_cache[t_id] = {"roll_no": roll_no, "sim": sim, "color": color, "last_updated": now_ts}
                                
                                # Attendance Logic
                                if roll_no not in self.last_scan_time or (now_ts - self.last_scan_time[roll_no] > 60):
                                    from app.services.attendance_service import mark_attendance
                                    asyncio.run_coroutine_threadsafe(mark_attendance(roll_no, "CCTV"), self.main_loop)
                                    self.last_scan_time[roll_no] = now_ts
                            else:
                                self.identity_cache[t_id] = {"roll_no": "Unknown", "sim": found_sim, "color": (0, 0, 255), "last_updated": now_ts}
                    else:
                        roll_no = identity["roll_no"]
                        sim = identity["sim"]
                        color = identity["color"]
                        
                    new_detections.append({"box": (int(x1), int(y1), int(x2), int(y2)), "text": roll_no, "sim": sim, "color": color})

                with self._det_lock:
                    self._current_detections = new_detections
                
                proc_count += 1
                if proc_count % 30 == 0:
                    p_fps = proc_count / (time.time() - start_time)
                    print(f"  [StreamManager] AI FPS: {p_fps:.2f}")

        except Exception as e:
            print(f"  [StreamManager] FATAL AI Error: {e}")
            traceback.print_exc()

    # ── MJPEG Broadcaster Thread (Enterprise Optimization) ───────────────────
    
    def _mjpeg_loop(self):
        """Dedicated thread to pre-encode MJPEG frames for all subscribers."""
        print("  [StreamManager] Starting MJPEG Broadcaster Thread")
        
        while not self._stop_event.is_set():
            try:
                # Wait for new frame from Capture Thread
                if not self._new_frame_event.wait(timeout=1.0):
                    continue
                self._new_frame_event.clear()
                
                frame = None
                with self._frame_lock:
                    if self._latest_frame is not None:
                        frame = self._latest_frame.copy()
                
                if frame is None:
                    continue
                
                # 1. Capture a local copy of detections
                with self._det_lock:
                    detections = self._current_detections.copy()
                
                # 2. Resize FIRST (720p HD)
                annotated = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                
                # 3. Scale & Draw Detections
                # Calculate scale factors (Skip if source resolution not yet captured)
                if self._source_w > 0 and self._source_h > 0:
                    scale_x = self.TARGET_WIDTH / self._source_w
                    scale_y = self.TARGET_HEIGHT / self._source_h
                    
                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        # Scale coordinates to 720p
                        sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                        sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
                        
                        color = det["color"]
                        cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), color, 2)
                        
                        label = f"{det['text']} ({det['sim']:.2f})"
                        cv2.putText(annotated, label, (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # ✅ STORE ANNOTATED FRAME (Zero-copy for WebRTC)
                with self._annotated_lock:
                    self._latest_annotated_frame = annotated

                # 3. Only Encode MJPEG if there are active legacy subscribers
                with self._sub_lock:
                    has_subs = len(self._subscribers) > 0

                if not has_subs:
                    continue # SKIP JPEG ENCODING - Wastes CPU during WebRTC use

                _, buffer = cv2.imencode('.jpg', annotated, self._JPEG_PARAMS)
                jpeg_bytes = buffer.tobytes()
                
                with self._jpeg_lock:
                    self._latest_jpeg = jpeg_bytes
                
                # 4. Notify all async subscribers (MJPEG legacy)
                if self.main_loop:
                    def safe_put(q):
                        try: q.put_nowait(True)
                        except asyncio.QueueFull: pass
                        
                    with self._sub_lock:
                        for q in self._subscribers:
                            self.main_loop.call_soon_threadsafe(safe_put, q)
                            
            except Exception as e:
                print(f"  [StreamManager] MJPEG Loop warning: {e}")
                time.sleep(0.1)

    # ── MJPEG Delivery ────────────────────────────────────────────────────────
    
    _JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    
    def _draw_annotations(self, frame):
        with self._det_lock:
            detections = self._current_detections.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = det["color"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det['text']} {det['sim']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    async def get_frame_generator(self):
        """Async generator with unique queue for each subscriber to prevent race conditions."""
        queue = asyncio.Queue(maxsize=1)
        with self._sub_lock:
            self._subscribers.append(queue)
        
        print(f"  [StreamManager] New subscriber connected (Total: {len(self._subscribers)})")
        
        try:
            while True:
                # Wait for next frame signal
                await queue.get()
                
                # Flush the queue to stay real-time
                while not queue.empty():
                    queue.get_nowait()
                
                # Fetch pre-computed JPEG buffer
                jpeg_bytes = None
                with self._jpeg_lock:
                    jpeg_bytes = self._latest_jpeg
                
                if jpeg_bytes:
                    # Enforce FPS cap
                    now = time.time()
                    elapsed = now - self._last_mjpeg_time
                    if elapsed < self._mjpeg_interval:
                        await asyncio.sleep(self._mjpeg_interval - elapsed)
                    self._last_mjpeg_time = time.time()
                    
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        finally:
            with self._sub_lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)
            print(f"  [StreamManager] Subscriber disconnected (Remaining: {len(self._subscribers)})")

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            if self._latest_frame is None: return None
            frame = self._latest_frame.copy()
        frame = self._draw_annotations(frame)
        _, buffer = cv2.imencode('.jpg', frame, self._JPEG_PARAMS)
        return buffer.tobytes()

streamer = EnterpriseStreamManager()
