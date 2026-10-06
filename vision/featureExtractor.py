# vision/featureExtractor.py
import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np

# Inicialização da rede neural para Re-ID (desativa a cabeça de classificação)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
_model.classifier = torch.nn.Identity()  # Remove a camada final para obter embeddings (576D)
_model.to(_device).eval()

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),  # Proporção padrão de Re-ID de pedestres (H x W)
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _crop_person(frame: np.ndarray, box: list) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img = frame.shape[:2]
    
    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(w_img, x2)
    y2_c = min(h_img, y2)
    
    return frame[y1_c:y2_c, x1_c:x2_c]

@torch.no_grad()
def extract_deep_reid_feature(frame: np.ndarray, box: list) -> np.ndarray:
    crop = _crop_person(frame, box)
    if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
        return np.zeros(576, dtype=np.float32)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = _transform(crop_rgb).unsqueeze(0).to(_device)

    feat = _model(tensor).squeeze(0).cpu().numpy()
    norm = np.linalg.norm(feat)
    
    return (feat / norm if norm > 0 else feat).astype(np.float32)

# Mapeia as chamadas usadas no worker para o extrator profundo
extract_color_histogram = extract_deep_reid_feature
extract_color_histogram_fast = extract_deep_reid_feature
extract_color_histogram_rich = extract_deep_reid_feature