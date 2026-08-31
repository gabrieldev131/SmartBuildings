# vision/featureExtractor.py
import cv2
import numpy as np


def _crop_person(frame: np.ndarray, box: list):
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

    return crop


def _histogram(crop: np.ndarray, crop_size, bins, empty_len) -> np.ndarray:
    if crop.size == 0:
        return np.zeros(empty_len)

    resized = cv2.resize(crop, crop_size, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


# ---------------------------------------------------------------------------
# VERSÃO BARATA: usada TODO frame, para TODA deteção -- alimenta o custo de
# aparência interno do DeepSORT (embeds/others). Precisa ser rápida, não
# precisa ser super discriminativa: o DeepSORT já tem posição/IOU/Kalman
# para ajudar na associação quadro-a-quadro dentro da MESMA câmara.
# ---------------------------------------------------------------------------
_FAST_CROP_SIZE = (48, 96)
_FAST_BINS = (8, 4, 4)      # 128 valores
_FAST_LEN = 8 * 4 * 4

def extract_color_histogram_fast(frame: np.ndarray, box: list) -> np.ndarray:
    crop = _crop_person(frame, box)
    return _histogram(crop, _FAST_CROP_SIZE, _FAST_BINS, _FAST_LEN)


# ---------------------------------------------------------------------------
# VERSÃO RICA: usada só nos momentos que realmente importam para o Re-ID
# global entre câmaras -- quando uma pessoa nova aparece numa câmara (decide
# se é alguém já visto antes) e nas atualizações periódicas (throttled) do
# vetor EMA guardado no GlobalIdentityManager. Isso acontece raramente por
# pessoa (não a cada frame), então pode pagar o custo de mais bins/mais
# resolução sem pesar na CPU de forma perceptível.
# ---------------------------------------------------------------------------
_RICH_CROP_SIZE = (96, 192)
_RICH_BINS = (16, 8, 8)     # 1024 valores -- qualidade original
_RICH_LEN = 16 * 8 * 8

def extract_color_histogram_rich(frame: np.ndarray, box: list) -> np.ndarray:
    crop = _crop_person(frame, box)
    return _histogram(crop, _RICH_CROP_SIZE, _RICH_BINS, _RICH_LEN)


# Mantido por compatibilidade com qualquer código antigo que ainda importe
# o nome original -- aponta para a versão rica (a de melhor qualidade),
# nunca para a barata, para não reintroduzir silenciosamente o mesmo bug.
#extract_color_histogram = extract_color_histogram_rich