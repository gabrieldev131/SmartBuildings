# vision/featureExtractor.py
"""
High-Performance Re-Identification Feature Extractor (GPU FP16 / CPU FP32).
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

_backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features.to(_device, dtype=_dtype).eval()
_backbone_mid = _backbone[:9]
_backbone_deep = _backbone[9:]

_mean = torch.tensor([0.485, 0.456, 0.406], device=_device, dtype=_dtype).view(1, 3, 1, 1)
_std = torch.tensor([0.229, 0.224, 0.225], device=_device, dtype=_dtype).view(1, 3, 1, 1)


def _compute_deep_part(deep_map: torch.Tensor) -> torch.Tensor:
    g_avg = 0.68 * deep_map.mean(dim=(2, 3)) + 0.32 * deep_map.amax(dim=(2, 3))
    s4_raw = 0.68 * deep_map.mean(dim=3) + 0.32 * deep_map.amax(dim=3)
    s4_pad = F.pad(s4_raw, (1, 1), mode="replicate")
    s4 = 0.65 * s4_raw + 0.175 * (s4_pad[:, :, :-2] + s4_pad[:, :, 2:])
    stripes = s4.view(-1, 4, 144, 4).mean(dim=1).permute(0, 2, 1).reshape(-1, 576)
    stripes = stripes - stripes.mean(dim=1, keepdim=True)
    g_centered = g_avg - g_avg.mean(dim=1, keepdim=True)
    return F.normalize(F.normalize(stripes, dim=1) + 0.55 * F.normalize(g_centered, dim=1), dim=1)


def _compute_mid_part(mid_map: torch.Tensor) -> torch.Tensor:
    mid_inner = 0.82 * mid_map[:, :, :, 1:3].mean(dim=3, keepdim=True) + 0.18 * mid_map.mean(dim=3, keepdim=True)
    mid_4 = F.adaptive_avg_pool2d(mid_inner, (4, 1)).flatten(1)
    mid_2 = F.adaptive_avg_pool2d(mid_inner, (2, 1)).repeat(1, 1, 2, 1).flatten(1)
    mid_center = 0.70 * mid_4 + 0.30 * mid_2
    return F.normalize(mid_center - mid_center.mean(dim=1, keepdim=True), dim=1)


def _build_color_channels(crops_rgb_01: torch.Tensor) -> torch.Tensor:
    body = crops_rgb_01[:, :, 14:114, 16:48].float()
    r, g, b = body[:, 0:1], body[:, 1:2], body[:, 2:3]
    ch_mean = body.mean(dim=(2, 3), keepdim=True).clamp(min=0.10, max=0.92)
    wb = (body * (ch_mean.mean(dim=1, keepdim=True) / ch_mean)).clamp(0.0, 1.0)
    rw, gw, bw = wb[:, 0:1], wb[:, 1:2], wb[:, 2:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lum_norm = ((lum - lum.mean(dim=(2, 3), keepdim=True)) / lum.std(dim=(2, 3), keepdim=True).clamp(min=0.06)).clamp(-2.5, 2.5) * 0.25 + 0.5
    sum_rgb, sum_wb = (r + g + b).clamp(min=1e-2), (rw + gw + bw).clamp(min=1e-2)
    rwn, gwn, bwn = rw / sum_wb, gw / sum_wb, bw / sum_wb
    cmax, cmin = torch.maximum(torch.maximum(rw, gw), bw), torch.minimum(torch.minimum(rw, gw), bw)
    return torch.cat([rw, gw, bw, lum_norm, (rw - gw) * 1.35, (0.5 * (rw + gw) - bw) * 1.35, r / sum_rgb, g / sum_rgb, b / sum_rgb, (cmax - cmin) / (cmax + 1e-2), rwn - gwn, bwn - 0.5 * (rwn + gwn)], dim=1)


def _compute_color_part(crops_rgb_01: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    col_stack = _build_color_channels(crops_rgb_01)
    col_core = 0.75 * col_stack[:, :, :, 8:24].mean(dim=3, keepdim=True) + 0.25 * col_stack.mean(dim=3, keepdim=True)
    col_avg_8 = F.adaptive_avg_pool2d(col_core, (8, 1)).flatten(1)
    col_coarse_4 = F.interpolate(F.adaptive_avg_pool2d(col_core, (4, 1)), size=(8, 1), mode="bilinear", align_corners=False).flatten(1)
    col_max_8 = F.adaptive_max_pool2d(col_core, (8, 1)).flatten(1)
    col_feat = torch.cat([0.65 * col_avg_8 + 0.35 * col_coarse_4, 0.65 * col_coarse_4 + 0.35 * col_max_8], dim=1)
    return F.normalize(col_feat - col_feat.mean(dim=1, keepdim=True), dim=1).to(target_dtype)


def extract_multistripe_embeddings_from_maps(crops_rgb_01: torch.Tensor, mid_map: torch.Tensor, deep_map: torch.Tensor) -> torch.Tensor:
    deep_part = _compute_deep_part(deep_map)
    mid_part = _compute_mid_part(mid_map)
    col_part = _compute_color_part(crops_rgb_01, deep_part.dtype)
    b1 = F.normalize(deep_part[:, :192] + 0.55 * deep_part[:, 192:384], dim=1)
    b2 = F.normalize(0.75 * deep_part[:, 384:] + 0.68 * mid_part, dim=1)
    return F.normalize(torch.cat([b1, b2, 0.95 * col_part], dim=1), dim=1)


@torch.inference_mode()
def extract_embeddings_from_crops(crops_rgb_01: torch.Tensor) -> torch.Tensor:
    if len(crops_rgb_01) == 0:
        return torch.empty((0, EMBEDDING_DIM), device=crops_rgb_01.device, dtype=torch.float32)
    crops_norm = (crops_rgb_01.to(_dtype) - _mean) / _std
    mid_map = _backbone_mid(crops_norm)
    deep_map = _backbone_deep(mid_map)
    return extract_multistripe_embeddings_from_maps(crops_rgb_01, mid_map, deep_map).float()


@torch.inference_mode()
def extract_features_batch_from_frame(frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    t_rgb = torch.from_numpy(frame).to(_device, non_blocking=_use_gpu).flip(-1).permute(2, 0, 1).unsqueeze(0).contiguous().to(_dtype).div_(255.0)
    boxes_t = torch.as_tensor(boxes_xyxy[:, :4], device=_device, dtype=_dtype)
    rois = torch.cat([torch.zeros((len(boxes_t), 1), device=_device, dtype=_dtype), boxes_t], dim=1)
    crops = roi_align(t_rgb, rois, output_size=(128, 64), spatial_scale=1.0, aligned=True)
    return extract_embeddings_from_crops(crops).cpu().numpy()


@torch.inference_mode()
def extract_deep_reid_feature(frame: np.ndarray, box: list) -> np.ndarray:
    feats = extract_features_batch_from_frame(frame, np.asarray([box[:4]], dtype=np.float32))
    return feats[0] if len(feats) > 0 else np.zeros(EMBEDDING_DIM, dtype=np.float32)
