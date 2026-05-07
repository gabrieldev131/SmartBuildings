# vision/feature_extractor.py
import cv2
import numpy as np

def extract_color_histogram(frame: np.ndarray, box: list) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(512)
        
    hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()