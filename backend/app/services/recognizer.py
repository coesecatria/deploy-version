"""
Face Recognition — Enterprise Stack using SCRFD + ArcFace + FAISS + TensorRT.
"""

import io
import cv2
import numpy as np
from PIL import Image

from app.services.face_engine import engine
from app.services.scrfd import SCRFD
from app.utils.alignment import align_face

# Initialize the custom SCRFD detector wrapper once
_detector = None

def _get_detector():
    global _detector
    if _detector is None and engine.det_model:
        _detector = SCRFD(engine.det_model)
    return _detector

def recognize(image_bytes: bytes) -> tuple:
    """Enterprise single-face recognition with alignment."""
    if not engine._initialized:
        return None, None, "Engine not initialized"

    try:
        # Convert bytes to CV2 image (BGR)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, "Invalid image data"

        detector = _get_detector()
        bboxes, scores, kpss = detector.detect(img, thresh=0.5)

        if bboxes is None or len(bboxes) == 0:
            return None, None, "No face detected"

        # Take the best face (highest score)
        idx = np.argmax(scores)
        face_aligned = align_face(img, kpss[idx])
        
        if face_aligned is None:
            return None, None, "Face alignment failed"

        # Generate embedding via ArcFace ONNX
        # Convert BGR to RGB for embedding
        face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        embedding = engine.get_embedding(face_rgb)
        
        # Search in FAISS
        embedding = np.expand_dims(embedding, axis=0).astype(np.float32)
        distances, indices = engine.index.search(embedding, k=1)
        
        similarity = float(distances[0][0])
        matched_idx = int(indices[0][0])
        
        if matched_idx < 0:
            return None, None, "Unknown"
            
        return engine.labels[matched_idx], similarity, None

    except Exception as e:
        return None, None, f"Recognition error: {str(e)}"

def recognize_multi(image_bytes: bytes) -> dict:
    """Enterprise multi-face recognition with batch alignment."""
    if not engine._initialized:
        return {"faces_detected": 0, "faces_recognized": 0, "results": [], "error": "Engine not initialized"}

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"faces_detected": 0, "faces_recognized": 0, "results": [], "error": "Invalid image"}

        detector = _get_detector()
        bboxes, scores, kpss = detector.detect(img, thresh=0.5)

        if bboxes is None or len(bboxes) == 0:
            return {"faces_detected": 0, "faces_recognized": 0, "results": [], "error": None}

        results = []
        best_matches = {}

        for i in range(len(bboxes)):
            face_aligned = align_face(img, kpss[i])
            if face_aligned is None: continue
            
            face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
            embedding = engine.get_embedding(face_rgb)
            
            # Vector search
            embedding = np.expand_dims(embedding, axis=0).astype(np.float32)
            distances, indices = engine.index.search(embedding, k=1)
            
            sim = float(distances[0][0])
            idx = int(indices[0][0])
            
            if idx >= 0:
                roll_no = engine.labels[idx]
                if roll_no not in best_matches or sim > best_matches[roll_no]:
                    best_matches[roll_no] = sim

        results = [{"roll_no": r, "similarity": round(s, 4)} for r, s in best_matches.items()]
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "faces_detected": len(bboxes),
            "faces_recognized": len(results),
            "results": results,
            "error": None
        }

    except Exception as e:
        return {"faces_detected": 0, "faces_recognized": 0, "results": [], "error": f"Multi-recognition error: {str(e)}"}
