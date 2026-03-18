import os
import pickle
import numpy as np
import faiss
import cv2
import onnxruntime as ort

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models", "insightface")

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "student_index.faiss")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pkl")

class FaceEngine:
    """Enterprise Face Recognition Engine using raw ONNX models with TensorRT."""

    def __init__(self):
        self.det_model = None
        self.rec_model = None
        self.index = None
        self.labels = None
        self._initialized = False

    def initialize(self):
        if self._initialized: return

        # ── 0. DLL Path Configuration (Windows Only) ──
        # Ensure we can find the CUDA/cuDNN DLLs inside the virtual environment
        import sys
        if sys.platform == "win32":
            venv_path = os.path.dirname(os.path.dirname(os.path.dirname(MODELS_DIR)))
            torch_lib = os.path.join(venv_path, "venv", "Lib", "site-packages", "torch", "lib")
            ort_capi = os.path.join(venv_path, "venv", "Lib", "site-packages", "onnxruntime", "capi")
            
            if os.path.exists(torch_lib):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    try: os.add_dll_directory(torch_lib)
                    except: pass
            
            if os.path.exists(ort_capi):
                os.environ["PATH"] = ort_capi + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    try: os.add_dll_directory(ort_capi)
                    except: pass

        # ── Execution Provider Selection ──
        # Order: TensorRT (fastest) -> CUDA -> CPU
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        print(f"  [FaceEngine] Initializing with providers: {providers}")

        # ── 1. Detection Model (SCRFD - det_10g.onnx) ──
        det_path = os.path.join(MODELS_DIR, "det_10g.onnx")
        self.det_model = ort.InferenceSession(det_path, providers=providers)
        print("  [FaceEngine] SCRFD Detection loaded (ONNX/TensorRT)")

        # ── 2. Recognition Model (ArcFace - w600k_r50.onnx) ──
        rec_path = os.path.join(MODELS_DIR, "w600k_r50.onnx")
        self.rec_model = ort.InferenceSession(rec_path, providers=providers)
        print("  [FaceEngine] ArcFace Embedding loaded (ONNX/TensorRT)")

        # ── 3. Search Infrastructure ──
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(LABELS_PATH, "rb") as f:
            self.labels = pickle.load(f)
        
        print(f"  [FaceEngine] FAISS system ready ({self.index.ntotal} records)")

        self._initialized = True
        print("  [FaceEngine] Enterprise stack online.")

    def get_embedding(self, aligned_face):
        """
        Extracts ArcFace embedding from a 112x112 aligned face.
        Expects RGB image with shape (112, 112, 3).
        """
        # Preprocessing: Transpose to (C, H, W) and normalize
        img = aligned_face.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0) # Batch size 1
        
        # Inference
        inputs = {self.rec_model.get_inputs()[0].name: img}
        outs = self.rec_model.run(None, inputs)
        feat = outs[0].flatten()
        # Unit normalization for Cosine/L2 consistency
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat

# Global singleton
engine = FaceEngine()
