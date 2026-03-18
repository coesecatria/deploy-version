"""
Student Registration & Deletion — FAISS index and photo management.
"""

import os
import io
import shutil
import pickle
import numpy as np
import cv2
import faiss
from PIL import Image

from app.services.face_engine import engine, BASE_DIR, FAISS_INDEX_PATH, LABELS_PATH
from app.services.scrfd import SCRFD
from app.utils.alignment import align_face

# Initialize detector once
_detector = None

def _get_detector():
    global _detector
    if _detector is None and engine.det_model:
        _detector = SCRFD(engine.det_model)
    return _detector

def register_faces(roll_no: str, image_bytes_list: list[bytes]) -> dict:
    """Enterprise registration using SCRFD + Aligned ArcFace."""
    if not engine._initialized:
        return {"success": False, "error": "Engine not initialized"}

    student_dir = os.path.join(BASE_DIR, "processed_dataset", roll_no)
    os.makedirs(student_dir, exist_ok=True)

    new_embeddings = []
    images_saved = 0

    detector = _get_detector()

    for i, img_bytes in enumerate(image_bytes_list):
        try:
            # Save original for record
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: continue

            save_path = os.path.join(student_dir, f"reg_{i}_orig.jpg")
            cv2.imwrite(save_path, img)
            images_saved += 1

            # Detect and Align
            bboxes, scores, kpss = detector.detect(img, thresh=0.5)
            if bboxes is None or len(bboxes) == 0:
                continue

            # Take the best face
            idx = np.argmax(scores)
            face_aligned = align_face(img, kpss[idx])
            if face_aligned is None: continue

            # Scale and Encode
            face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
            embedding = engine.get_embedding(face_rgb)
            
            # ArcFace embeddings are already normalized in my get_embedding, 
            # but let's be explicitly safe for FAISS IndexFlatIP
            norm = np.linalg.norm(embedding)
            if norm == 0: continue
            embedding = (embedding / norm).astype(np.float32)
            
            new_embeddings.append(embedding)

        except Exception as e:
            print(f"  [Registrar] Frame {i} failed: {e}")
            continue

    if not new_embeddings:
        return {
            "success": False,
            "embeddings_added": 0,
            "total_images": len(image_bytes_list),
            "images_saved": images_saved,
            "error": "No faces detected in any of the captured images",
        }

    # Bulk add to FAISS
    embeddings_matrix = np.vstack(new_embeddings).astype(np.float32)
    engine.index.add(embeddings_matrix)

    for _ in new_embeddings:
        engine.labels.append(roll_no)

    faiss.write_index(engine.index, FAISS_INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(engine.labels, f)

    return {
        "success": True,
        "embeddings_added": len(new_embeddings),
        "total_images": len(image_bytes_list),
        "images_saved": images_saved,
        "error": None,
    }

def delete_student(roll_no: str) -> dict:
    """Delete a student from FAISS index and local storage."""
    if not engine._initialized:
        return {"success": False, "error": "Engine not initialized"}

    roll_no = roll_no.upper()
    indices_to_keep = [i for i, label in enumerate(engine.labels) if label != roll_no]
    removed_count = len(engine.labels) - len(indices_to_keep)

    if removed_count == 0 and not os.path.exists(os.path.join(BASE_DIR, "processed_dataset", roll_no)):
        return {"success": False, "error": f"No data found for {roll_no}"}

    if indices_to_keep and engine.index.ntotal > 0:
        all_embeddings = np.array([engine.index.reconstruct(i) for i in indices_to_keep], dtype=np.float32)
        new_labels = [engine.labels[i] for i in indices_to_keep]
        # ArcFace uses 512 dimensions
        new_index = faiss.IndexFlatIP(512)
        new_index.add(all_embeddings)
        engine.index = new_index
        engine.labels = new_labels
    else:
        engine.index = faiss.IndexFlatIP(512)
        engine.labels = []

    faiss.write_index(engine.index, FAISS_INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(engine.labels, f)

    # Clean up photos
    student_dir = os.path.join(BASE_DIR, "processed_dataset", roll_no)
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir)

    return {
        "success": True,
        "embeddings_removed": removed_count,
        "index_total": engine.index.ntotal,
    }
