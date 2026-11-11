# vision/featureExtractor.py
"""
High-Performance Re-Identification Feature Extractor (GPU FP16 / CPU FP32).
Combines MobileNetV3-Small Mid-Level Texture Stripes + Deep Semantic Horizontal Stripes
with 8-Stripe 12-Channel Opponent Color, Luminance, Saturation & Chromaticity descriptors.
Respects Config.USE_GPU for seamless GPU or CPU execution.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import roi_align

from Config import Config

_use_gpu_cfg = bool(getattr(Config, "USE_GPU", True))
_device = torch.device("cuda:0" if (_use_gpu_cfg and torch.cuda.is_available()) else "cpu")
_use_fp16 = _device.type == "cuda"

_backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features
_backbone.to(_device).eval()
if _use_fp16:
    _backbone.half()

_backbone_mid = _backbone[:9]
_backbone_deep = _backbone[9:]

_dtype = torch.float16 if _use_fp16 else torch.float32
_mean = torch.tensor([0.485, 0.456, 0.406], device=_device, dtype=_dtype).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225], device=_device, dtype=_dtype).view(1, 3, 1, 1)

# Pre-warm common batch buckets on GPU to avoid cuDNN runtime autotuning stalls
_BUCKETS = (1, 4, 8, 16, 24, 32, 48, 64, 96, 128)
_pad_buf = torch.zeros((128, 3, 128, 64), device=_device, dtype=_dtype) if _use_fp16 else None
if _use_fp16:
    with torch.inference_mode():
        for _b in _BUCKETS:
            _ = _backbone(_pad_buf[:_b])
    torch.cuda.synchronize()


def extract_multistripe_embeddings_from_maps(
    crops_rgb_01: torch.Tensor,
    mid_map: torch.Tensor,
    deep_map: torch.Tensor,
) -> torch.Tensor:
    """
    Core tensor graph computing centered 576D L2-normalized embeddings from:
    - deep_map: (N, 576, 4, 2) high-level semantic horizontal stripes
    - mid_map:  (N, 48, 8, 4) mid-level clothing texture/pattern horizontal stripes
    - crops_rgb_01: (N, 3, 128, 64) 8-stripe 12-channel opponent color, luminance & chromaticity
    Shared identically by BatchInferenceEngine (CUDA Graph & CPU) and standalone extractors.
    """
    # 1. Deep horizontal stripes + centered global context (576D)
    g_avg = deep_map.mean(dim=(2, 3))  # (N, 576)
    s4 = 0.70 * deep_map.mean(dim=3) + 0.30 * deep_map.amax(dim=3)  # (N, 576, 4)
    s4_grouped = s4.view(-1, 4, 144, 4).mean(dim=1)  # (N, 144, 4)
    deep_stripes = s4_grouped.permute(0, 2, 1).reshape(-1, 576)
    deep_stripes = deep_stripes - deep_stripes.mean(dim=1, keepdim=True)
    g_centered = g_avg - g_avg.mean(dim=1, keepdim=True)
    deep_part = F.normalize(deep_stripes, dim=1) + 0.45 * F.normalize(g_centered, dim=1)
    deep_part = F.normalize(deep_part, dim=1)

    # 2. Mid-level texture horizontal stripes on center body (4 stripes x 48ch = 192D)
    mid_center = F.adaptive_avg_pool2d(mid_map[:, :, :, 1:3], (4, 1)).flatten(1)  # (N, 192)
    mid_center = mid_center - mid_center.mean(dim=1, keepdim=True)
    mid_part = F.normalize(mid_center, dim=1)

    # 3. 8-Stripe 12-Channel Color + Luminance + Opponent Chromaticity + Saturation on inner torso/legs
    body = crops_rgb_01[:, :, 12:116, 16:48].float()
    r, g, b = body[:, 0:1], body[:, 1:2], body[:, 2:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    rg = (r - g) * 1.4
    yb = (0.5 * (r + g) - b) * 1.4
    sum_rgb = (r + g + b).clamp(min=1e-2)
    rn = r / sum_rgb
    gn = g / sum_rgb
    bn = b / sum_rgb
    cmax = torch.maximum(torch.maximum(r, g), b)
    cmin = torch.minimum(torch.minimum(r, g), b)
    sat = (cmax - cmin) / (cmax + 1e-2)

    col_stack = torch.cat([r, g, b, lum, rg, yb, rn, gn, bn, sat, r - lum, b - lum], dim=1)  # 12 channels
    col_avg = F.adaptive_avg_pool2d(col_stack, (8, 1)).flatten(1)  # (N, 96)
    col_max = F.adaptive_max_pool2d(col_stack, (8, 1)).flatten(1)  # (N, 96)
    col_feat = torch.cat([col_avg, 0.70 * col_max], dim=1)  # (N, 192)
    col_feat = col_feat - col_feat.mean(dim=1, keepdim=True)
    col_part = F.normalize(col_feat, dim=1).to(deep_part.dtype)

    # Fuse into 3 orthogonal 192D blocks = 576D L2-normalized embedding
    b1 = F.normalize(deep_part[:, :192] + 0.50 * deep_part[:, 192:384], dim=1)
    b2 = F.normalize(0.75 * deep_part[:, 384:] + 0.65 * mid_part, dim=1)
    out = torch.cat([b1, b2, 1.15 * col_part], dim=1)
    return F.normalize(out, dim=1)


@torch.inference_mode()
def compute_part_based_embeddings(crops_rgb_01: torch.Tensor) -> torch.Tensor:
    """
    Given RGB crops in [0, 1] of shape (N, 3, 128, 64) on _device:
    Returns L2-normalized (N, 576) multi-stripe embeddings.
    """
    n = crops_rgb_01.shape[0]
    if n == 0:
        return torch.empty((0, 576), device=crops_rgb_01.device, dtype=torch.float32)

    crops_norm = (crops_rgb_01.to(_dtype) - _mean) / _std

    if _use_fp16 and _pad_buf is not None and n <= 128 and crops_norm.device == _device:
        bucket = next((b for b in _BUCKETS if b >= n), n)
        if bucket > n:
            _pad_buf[:n].copy_(crops_norm)
            mid_map = _backbone_mid(_pad_buf[:bucket])[:n]
        else:
            mid_map = _backbone_mid(crops_norm)
    else:
        mid_map = _backbone_mid(crops_norm)

    deep_map = _backbone_deep(mid_map)
    return extract_multistripe_embeddings_from_maps(crops_rgb_01, mid_map, deep_map)


@torch.inference_mode()
def extract_features_batch_from_frame(frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Extract 576D L2-normalized embeddings for all bounding boxes in a single frame in one batch.
    """
    if len(boxes_xyxy) == 0:
        return np.empty((0, 576), dtype=np.float32)

    t_bgr = torch.from_numpy(frame).to(_device, non_blocking=_use_fp16)
    t_rgb = t_bgr.flip(-1).permute(2, 0, 1).unsqueeze(0).contiguous().to(_dtype).div_(255.0)

    boxes_t = torch.as_tensor(boxes_xyxy, device=_device, dtype=_dtype)
    batch_col = torch.zeros((len(boxes_xyxy), 1), device=_device, dtype=_dtype)
    rois = torch.cat([batch_col, boxes_t[:, :4]], dim=1)

    crops = roi_align(t_rgb, rois, output_size=(128, 64), spatial_scale=1.0, aligned=True)
    feats = compute_part_based_embeddings(crops)
    return feats.float().cpu().numpy()


@torch.inference_mode()
def extract_deep_reid_feature(frame: np.ndarray, box: list) -> np.ndarray:
    """
    Extract a single 576D L2-normalized embedding for one bounding box.
    """
    x1, y1, x2, y2 = map(int, box[:4])
    h_img, w_img = frame.shape[:2]
    x1_c, y1_c = max(0, x1), max(0, y1)
    x2_c, y2_c = min(w_img, x2), min(h_img, y2)

    if (x2_c - x1_c) < 10 or (y2_c - y1_c) < 15:
        return np.zeros(576, dtype=np.float32)

    crop = frame[y1_c:y2_c, x1_c:x2_c]
    t_bgr = torch.from_numpy(crop).to(_device)
    t_rgb = t_bgr.flip(-1).permute(2, 0, 1).unsqueeze(0).contiguous().to(_dtype).div_(255.0)
    t_resized = F.interpolate(t_rgb, size=(128, 64), mode="bilinear", align_corners=False)

    feat = compute_part_based_embeddings(t_resized)
    return feat[0].float().cpu().numpy()


# Backward-compatible aliases
extract_color_histogram = extract_deep_reid_feature
extract_color_histogram_fast = extract_deep_reid_feature
extract_color_histogram_rich = extract_deep_reid_feature
