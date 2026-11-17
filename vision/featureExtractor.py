# vision/featureExtractor.py
"""
High-Performance Re-Identification Feature Extractor (GPU FP16 / CPU FP32).
Uses Torchvision's `mobilenet_v3_small` and `roi_align` to compute 576D L2-normalized
embeddings combining deep semantic stripes, mid-level texture stripes, and
illumination-invariant (Gray-World white-balanced) color descriptors.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import roi_align

from Config import Config

EMBEDDING_DIM = 576

_use_gpu = bool(getattr(Config, "USE_GPU", True)) and torch.cuda.is_available()
_device = torch.device("cuda:0" if _use_gpu else "cpu")
_dtype = torch.float16 if _use_gpu else torch.float32

_backbone = (
    models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    .features.to(_device, dtype=_dtype)
    .eval()
)
_backbone_mid = _backbone[:9]
_backbone_deep = _backbone[9:]

_mean = torch.tensor([0.485, 0.456, 0.406], device=_device, dtype=_dtype).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225], device=_device, dtype=_dtype).view(1, 3, 1, 1)


def extract_multistripe_embeddings_from_maps(
    crops_rgb_01: torch.Tensor,
    mid_map: torch.Tensor,
    deep_map: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 576D L2-normalized appearance embeddings from:
    - deep_map: (N, 576, 4, 2) high-level semantic feature maps
    - mid_map:  (N, 48, 8, 4) mid-level clothing texture feature maps
    - crops_rgb_01: (N, 3, 128, 64) RGB person crops in [0, 1]
    """
    # 1. Deep semantic horizontal stripes + global context (576D)
    g_avg = 0.68 * deep_map.mean(dim=(2, 3)) + 0.32 * deep_map.amax(dim=(2, 3))
    s4_raw = 0.68 * deep_map.mean(dim=3) + 0.32 * deep_map.amax(dim=3)
    s4_pad = F.pad(s4_raw, (1, 1), mode="replicate")
    s4 = 0.65 * s4_raw + 0.175 * (s4_pad[:, :, :-2] + s4_pad[:, :, 2:])

    deep_stripes = s4.view(-1, 4, 144, 4).mean(dim=1).permute(0, 2, 1).reshape(-1, 576)
    deep_stripes = deep_stripes - deep_stripes.mean(dim=1, keepdim=True)
    g_centered = g_avg - g_avg.mean(dim=1, keepdim=True)
    deep_part = F.normalize(F.normalize(deep_stripes, dim=1) + 0.55 * F.normalize(g_centered, dim=1), dim=1)

    # 2. Mid-level clothing texture horizontal stripes on inner torso (192D)
    mid_inner = 0.82 * mid_map[:, :, :, 1:3].mean(dim=3, keepdim=True) + 0.18 * mid_map.mean(dim=3, keepdim=True)
    mid_4 = F.adaptive_avg_pool2d(mid_inner, (4, 1)).flatten(1)
    mid_2 = F.adaptive_avg_pool2d(mid_inner, (2, 1)).repeat(1, 1, 2, 1).flatten(1)
    mid_center = 0.70 * mid_4 + 0.30 * mid_2
    mid_part = F.normalize(mid_center - mid_center.mean(dim=1, keepdim=True), dim=1)

    # 3. Illumination-invariant Gray-World white-balanced color & chromaticity stripes (192D)
    body = crops_rgb_01[:, :, 14:114, 16:48].float()
    r_raw, g_raw, b_raw = body[:, 0:1], body[:, 1:2], body[:, 2:3]

    ch_mean = body.mean(dim=(2, 3), keepdim=True).clamp(min=0.10, max=0.92)
    wb_body = (body * (ch_mean.mean(dim=1, keepdim=True) / ch_mean)).clamp(0.0, 1.0)
    rw, gw, bw = wb_body[:, 0:1], wb_body[:, 1:2], wb_body[:, 2:3]

    lum = 0.299 * r_raw + 0.587 * g_raw + 0.114 * b_raw
    lum_std = lum.std(dim=(2, 3), keepdim=True).clamp(min=0.06)
    lum_norm = ((lum - lum.mean(dim=(2, 3), keepdim=True)) / lum_std).clamp(-2.5, 2.5) * 0.25 + 0.5

    rg = (rw - gw) * 1.35
    yb = (0.5 * (rw + gw) - bw) * 1.35
    sum_rgb = (r_raw + g_raw + b_raw).clamp(min=1e-2)
    sum_wb = (rw + gw + bw).clamp(min=1e-2)
    rwn, gwn, bwn = rw / sum_wb, gw / sum_wb, bw / sum_wb
    cmax = torch.maximum(torch.maximum(rw, gw), bw)
    cmin = torch.minimum(torch.minimum(rw, gw), bw)
    sat = (cmax - cmin) / (cmax + 1e-2)

    col_stack = torch.cat(
        [
            rw, gw, bw, lum_norm, rg, yb,
            r_raw / sum_rgb, g_raw / sum_rgb, b_raw / sum_rgb,
            sat, rwn - gwn, bwn - 0.5 * (rwn + gwn),
        ],
        dim=1,
    )
    col_core = 0.75 * col_stack[:, :, :, 8:24].mean(dim=3, keepdim=True) + 0.25 * col_stack.mean(dim=3, keepdim=True)
    col_avg_8 = F.adaptive_avg_pool2d(col_core, (8, 1)).flatten(1)
    col_coarse_4 = F.interpolate(
        F.adaptive_avg_pool2d(col_core, (4, 1)), size=(8, 1), mode="bilinear", align_corners=False
    ).flatten(1)
    col_max_8 = F.adaptive_max_pool2d(col_core, (8, 1)).flatten(1)

    col_feat = torch.cat([0.65 * col_avg_8 + 0.35 * col_coarse_4, 0.65 * col_coarse_4 + 0.35 * col_max_8], dim=1)
    col_part = F.normalize(col_feat - col_feat.mean(dim=1, keepdim=True), dim=1).to(deep_part.dtype)

    # Fuse into 3 balanced 192D blocks = 576D L2-normalized embedding
    b1 = F.normalize(deep_part[:, :192] + 0.55 * deep_part[:, 192:384], dim=1)
    b2 = F.normalize(0.75 * deep_part[:, 384:] + 0.68 * mid_part, dim=1)
    return F.normalize(torch.cat([b1, b2, 0.95 * col_part], dim=1), dim=1)


@torch.inference_mode()
def extract_embeddings_from_crops(crops_rgb_01: torch.Tensor) -> torch.Tensor:
    """Extract 576D L2-normalized embeddings from RGB crops of shape (N, 3, 128, 64) in [0, 1]."""
    if len(crops_rgb_01) == 0:
        return torch.empty((0, EMBEDDING_DIM), device=crops_rgb_01.device, dtype=torch.float32)

    crops_norm = (crops_rgb_01.to(_dtype) - _mean) / _std
    mid_map = _backbone_mid(crops_norm)
    deep_map = _backbone_deep(mid_map)
    return extract_multistripe_embeddings_from_maps(crops_rgb_01, mid_map, deep_map).float()


@torch.inference_mode()
def extract_features_batch_from_frame(frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Extract 576D L2-normalized embeddings for all bounding boxes in a single BGR frame."""
    if len(boxes_xyxy) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    t_bgr = torch.from_numpy(frame).to(_device, non_blocking=_use_gpu)
    t_rgb = t_bgr.flip(-1).permute(2, 0, 1).unsqueeze(0).contiguous().to(_dtype).div_(255.0)
    boxes_t = torch.as_tensor(boxes_xyxy[:, :4], device=_device, dtype=_dtype)
    rois = torch.cat([torch.zeros((len(boxes_t), 1), device=_device, dtype=_dtype), boxes_t], dim=1)

    crops = roi_align(t_rgb, rois, output_size=(128, 64), spatial_scale=1.0, aligned=True)
    return extract_embeddings_from_crops(crops).cpu().numpy()


@torch.inference_mode()
def extract_deep_reid_feature(frame: np.ndarray, box: list) -> np.ndarray:
    """Extract a single 576D L2-normalized embedding for one [x1, y1, x2, y2] bounding box."""
    boxes = np.asarray([box[:4]], dtype=np.float32)
    feats = extract_features_batch_from_frame(frame, boxes)
    return feats[0] if len(feats) > 0 else np.zeros(EMBEDDING_DIM, dtype=np.float32)
