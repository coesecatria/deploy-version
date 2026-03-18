import numpy as np
import cv2
from skimage import transform as trans

# Standard InsightFace reference points for 112x112 target size
SRC_PTS = np.array([
    [30.2946, 51.6963],  # Left eye
    [65.5318, 51.5014],  # Right eye
    [48.0252, 71.7366],  # Nose tip
    [33.5493, 92.3655],  # Left mouth corner
    [62.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)

def align_face(img, landmarks, border_value=0):
    """
    Aligns a face based on 5-point landmarks using a similarity transformation.
    Target size is 112x112 (InsightFace standard).
    """
    if landmarks is None or len(landmarks) != 5:
        return None
        
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, SRC_PTS)
    M = tform.params[0:2, :]
    
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=border_value)
    return aligned
