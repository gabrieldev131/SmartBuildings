# vision/featureExtractor.py
import cv2
import numpy as np

def extract_color_histogram(frame: np.ndarray, box: list) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    
    h, w = y2 - y1, x2 - x1
    y1_inner = max(0, int(y1 + h * 0.10))
    y2_inner = min(frame.shape[0], int(y2 - h * 0.10))
    x1_inner = max(0, int(x1 + w * 0.25))
    x2_inner = min(frame.shape[1], int(x2 - w * 0.25))
    
    crop = frame[y1_inner:y2_inner, x1_inner:x2_inner]
    
    if crop.size == 0:
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(1024)
            
    # MUDANÇA 3: Suavização (Gaussian Blur)
    # Ao aplicar um pequeno blur, nós diluímos pequenos detalhes de alto contraste 
    # (como um logotipo numa t-shirt que fica invisível se a pessoa se virar ligeiramente).
    # O histograma passará a concentrar-se nas cores globais dominantes da roupa.
    blurred_crop = cv2.GaussianBlur(crop, (5, 5), 0)
            
    hsv_crop = cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist([hsv_crop], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    
    return hist.flatten()