import os
import cv2
import numpy as np
import faiss
import pickle
from tqdm import tqdm

# Imports from our new engine
from app.services.face_engine import engine
from app.services.scrfd import SCRFD
from app.utils.alignment import align_face
from app.services.face_engine import FAISS_INDEX_PATH, LABELS_PATH, BASE_DIR

DATASET_DIR = os.path.join(BASE_DIR, "processed_dataset")

def main():
    print("--- Enterprise Data Migration: Re-embedding Dataset ---")
    
    # Initialize Engine
    engine.initialize()
    detector = SCRFD(engine.det_model)
    
    all_embeddings = []
    all_labels = []
    
    student_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    print(f"Found {len(student_folders)} students.")

    for roll_no in tqdm(student_folders, desc="Processing Students"):
        student_path = os.path.join(DATASET_DIR, roll_no)
        images = [f for f in os.listdir(student_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(student_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Use SCRFD to find face and landmarks
            bboxes, scores, kpss = detector.detect(img, thresh=0.5)
            
            if bboxes is not None and len(bboxes) > 0:
                # Take highest scoring face
                idx = np.argmax(scores)
                
                # Align perfectly for ArcFace
                face_aligned = align_face(img, kpss[idx])
                if face_aligned is None: continue
                
                # Get Embedding
                face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
                embedding = engine.get_embedding(face_rgb)
                
                # Normalize for FAISS IndexFlatIP
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    all_embeddings.append(embedding)
                    all_labels.append(roll_no)

    if not all_embeddings:
        print("Error: No embeddings generated. Check your images/models.")
        return

    # Create new index (ArcFace = 512 dims)
    print(f"\nMigration complete. Generated {len(all_embeddings)} embeddings.")
    
    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    new_index = faiss.IndexFlatIP(512)
    new_index.add(embeddings_matrix)
    
    # Persist
    faiss.write_index(new_index, FAISS_INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(all_labels, f)
        
    print(f"Saved new index to: {FAISS_INDEX_PATH}")
    print(f"Saved new labels to: {LABELS_PATH}")
    print("System is now fully upgraded and synchronized.")

if __name__ == "__main__":
    main()
