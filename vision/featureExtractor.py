# vision/featureExtractor.py
import cv2
import numpy as np

# Tamanho fixo do crop antes de calcular o histograma: assim o custo de
# cvtColor/GaussianBlur/calcHist é CONSTANTE por deteção, independente do
# tamanho da bounding box (uma pessoa perto da câmara não deve custar mais
# CPU que uma pessoa longe).
_HIST_CROP_SIZE = (48, 96)  # (largura, altura)


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
            return np.zeros(128)

    # Redimensiona para tamanho fixo pequeno -> custo de CPU previsível
    # e baixo, independente da distância da pessoa à câmara.
    crop = cv2.resize(crop, _HIST_CROP_SIZE, interpolation=cv2.INTER_AREA)

    blurred_crop = cv2.GaussianBlur(crop, (3, 3), 0)
    hsv_crop = cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2HSV)

    # Menos bins (8x4x4=128 em vez de 16x8x8=1024): histograma mais barato
    # de calcular, normalizar e comparar (np.dot), com perda desprezível de
    # poder discriminativo para roupas/cores dominantes.
    hist = cv2.calcHist([hsv_crop], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist.flatten()