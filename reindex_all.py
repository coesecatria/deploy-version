import os
import sys
import numpy as np
import cv2
import faiss
import pickle
import time

# Add backend to path so we can import app modules
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.services.face_engine import engine, FAISS_INDEX_PATH, LABELS_PATH, BASE_DIR
from app.services.registrar import _get_detector
from app.utils.alignment import align_face

def reindex_all():
    print("\n" + "="*60)
    print("  🚀 ATTEND-AI: FULL DATABASE RE-INDEX")
    print("="*60)
    
    # 1. Initialize Engine
    print("  [Step 1/4] Initializing Face Engine on GPU...")
    engine.initialize()
    detector = _get_detector()
    
    # 2. Reset Index and Labels
    print("  [Step 2/4] Resetting FAISS Index and Label Map...")
    # ArcFace uses 512-dim vectors
    engine.index = faiss.IndexFlatIP(512)
    engine.labels = []
    
    dataset_dir = os.path.join(BASE_DIR, "processed_dataset")
    if not os.path.exists(dataset_dir):
        print(f"  [Error] Dataset directory not found: {dataset_dir}")
        return

    student_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"  [Found] {len(student_folders)} students in dataset.")
    
    start_time = time.time()
    total_processed = 0
    total_embeddings = 0

    # 3. Process Folders
    print("  [Step 3/4] Aligning and Embedding Faces...")
    for roll_no in student_folders:
        student_path = os.path.join(dataset_dir, roll_no)
        # Use only original photos for re-indexing (ignoring any existing aligned ones if they existed)
        images = [f for f in os.listdir(student_path) if f.endswith(".jpg")]
        
        print(f"    → Processing {roll_no} ({len(images)} images)...", end=" ", flush=True)
        
        student_embeddings = []
        for img_name in images:
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Detect and Align
            bboxes, scores, kpss = detector.detect(img, thresh=0.5)
            if bboxes is None or len(bboxes) == 0:
                continue
                
            # Take the best face
            idx = np.argmax(scores)
            face_aligned = align_face(img, kpss[idx])
            if face_aligned is None: continue
            
            # Encode
            face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
            embedding = engine.get_embedding(face_rgb)
            
            # Normalize for FAISS
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (embedding / norm).astype(np.float32)
                student_embeddings.append(embedding)
        
        if student_embeddings:
            embeddings_matrix = np.vstack(student_embeddings).astype(np.float32)
            engine.index.add(embeddings_matrix)
            for _ in student_embeddings:
                engine.labels.append(roll_no)
            
            total_processed += 1
            total_embeddings += len(student_embeddings)
            print(f"Done. (+{len(student_embeddings)} embeddings)")
        else:
            print("Failed (No faces detected).")

    # 4. Save Results
    print("  [Step 4/4] Saving Optimized Database...")
    faiss.write_index(engine.index, FAISS_INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(engine.labels, f)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("="*60)
    print(f"  🎉 RE-INDEX COMPLETE!")
    print(f"  Students Processed: {total_processed}")
    print(f"  Total Embeddings:   {total_embeddings}")
    print(f"  Time Elapsed:       {duration:.2f} seconds")
    print("="*60 + "\n")

if __name__ == "__main__":
    reindex_all()
