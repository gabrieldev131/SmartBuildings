# vision/featureExtractor.py
"""
High-Performance GPU Re-Identification Feature Extractor.
Combines MobileNetV3-Small FP16 deep features with Part-Based Horizontal Stripe Pooling
(upper/lower body) and illumination-invariant chromaticity statistics.
Eliminates CPU/PIL bottlenecks via direct GPU tensor operations and ROI-Align.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import roi_align

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_use_fp16 = _device.type == "cuda"

_backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features
_backbone.to(_device).eval()
if _use_fp16:
    _backbone.half()

_dtype = torch.float16 if _use_fp16 else torch.float32
_mean = torch.tensor([0.485, 0.456, 0.406], device=_device, dtype=_dtype).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225], device=_device, dtype=_dtype).view(1, 3, 1, 1)

# Pre-warm common batch buckets on GPU to avoid cuDNN runtime autotuning stalls
_BUCKETS = (1, 4, 8, 16, 24, 32, 48, 64, 96, 128)
_pad_buf = torch.zeros((128, 3, 128, 64), device=_device, dtype=_dtype)
if _use_fp16:
    with torch.inference_mode():
        for _b in _BUCKETS:
            _ = _backbone(_pad_buf[:_b])
    torch.cuda.synchronize()


@torch.inference_mode()
def compute_part_based_embeddings(crops_rgb_01: torch.Tensor) -> torch.Tensor:
    """
    Given RGB crops in [0, 1] of shape (N, 3, 128, 64) on _device:
    1) Extracts MobileNetV3 feature map (N, 576, 4, 2)
    2) Pools Global Average (576D) + Upper/Lower Body Horizontal Stripes (2 x 576D)
    3) Injects illumination-normalized chromaticity (upper/lower torso-leg color ratios)
    Returns L2-normalized (N, 576) embeddings.
    """
    n = crops_rgb_01.shape[0]
    if n == 0:
        return torch.empty((0, 576), device=_device, dtype=torch.float32)

    crops_norm = (crops_rgb_01.to(_dtype) - _mean) / _std

    # Use bucketed forward pass on CUDA to prevent dynamic-batch cuDNN autotuning spikes
    if _use_fp16 and n <= 128:
        bucket = next((b for b in _BUCKETS if b >= n), n)
        if bucket > n:
            _pad_buf[:n].copy_(crops_norm)
            feat_map = _backbone(_pad_buf[:bucket])[:n]
        else:
            feat_map = _backbone(crops_norm)
    else:
        feat_map = _backbone(crops_norm)

    # Global representation + 2 horizontal stripes (torso vs legs)
    global_pool = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)
    stripes_pool = F.adaptive_max_pool2d(feat_map, (2, 1)).flatten(1)
    upper_stripe = stripes_pool[:, :576]
    lower_stripe = stripes_pool[:, 576:]

    fused = F.normalize(global_pool, dim=1) + 0.45 * F.normalize(upper_stripe, dim=1) + 0.45 * F.normalize(lower_stripe, dim=1)

    # Illumination-invariant chromaticity from center-cropped torso/legs (suppresses background edges)
    center_crop = crops_rgb_01[:, :, 16:112, 12:52].to(_dtype)
    lum = center_crop.sum(dim=1, keepdim=True).clamp_(min=1e-3)
    chrom = center_crop / lum  # Normalized rg-chromaticity invariant to brightness changes
    chrom_stripes = F.adaptive_avg_pool2d(chrom, (4, 2)).flatten(1)  # (N, 24)
    chrom_std = F.adaptive_max_pool2d(chrom, (4, 2)).flatten(1)     # (N, 24)
    color_feat = F.normalize(torch.cat([chrom_stripes, chrom_std], dim=1), dim=1)  # (N, 48)

    # Blend chromaticity signature into the first 48 dimensions of the embedding
    fused = F.normalize(fused, dim=1)
    fused[:, :48] = fused[:, :48] + 0.35 * color_feat
    return F.normalize(fused, dim=1)


@torch.inference_mode()
def extract_features_batch_from_frame(frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """
    Extract 576D L2-normalized embeddings for all bounding boxes in a single frame in one GPU batch.
    """
    if len(boxes_xyxy) == 0:
        return np.empty((0, 576), dtype=np.float32)

    t_bgr = torch.from_numpy(frame).to(_device, non_blocking=True)
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
