import onnxruntime as ort
import os

# Backend directory is where the script is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "insightface")

def main():
    det_path = os.path.join(MODELS_DIR, "det_10g.onnx")
    rec_path = os.path.join(MODELS_DIR, "w600k_r50.onnx")
    
    for name, path in [("Detection", det_path), ("Recognition", rec_path)]:
        print(f"\nModel: {name}")
        session = ort.InferenceSession(path)
        for input in session.get_inputs():
            print(f" Input: {input.name}, Shape: {input.shape}, Type: {input.type}")
        for output in session.get_outputs():
            print(f" Output: {output.name}, Shape: {output.shape}, Type: {output.type}")

if __name__ == "__main__":
    main()
