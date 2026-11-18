# vision/BatchInferenceEngine.py
"""
Batched Inference Engine with GPU (FP16 + CUDA Graphs) and CPU (FP32) support.
"""
import logging
import threading
import numpy as np
import torch
import torchvision.models as models
from torchvision.ops import box_iou, nms, roi_align
from ultralytics import YOLO

from vision.featureExtractor import EMBEDDING_DIM, extract_multistripe_embeddings_from_maps
from vision.GraphSlot import GraphSlot


class BatchInferenceEngine:
    """Centralized multi-camera person detection and Re-ID feature extraction engine (SRP)."""

    def __init__(self, config):
        self.config = config
        self.width, self.height = getattr(config, "PROCESSING_WIDTH", 640), getattr(config, "PROCESSING_HEIGHT", 480)
        self.max_dets = getattr(config, "MAX_DETECTIONS_PER_CAMERA", 18)
        self.use_gpu = bool(getattr(config, "USE_GPU", True)) and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        self.use_cuda_graphs = bool(getattr(config, "USE_CUDA_GRAPHS", True)) and self.use_gpu
        self._dtype = torch.float16 if self.use_gpu else torch.float32
        self.conf_low, self.iou_thresh = getattr(config, "YOLO_CONF_LOW", 0.35), getattr(config, "YOLO_IOU_THRESHOLD", 0.50)
        self.min_w, self.min_h = float(getattr(config, "MIN_BOX_WIDTH", 16)), float(getattr(config, "MIN_BOX_HEIGHT", 34))
        self.min_ar, self.max_ar = float(getattr(config, "MIN_ASPECT_RATIO", 0.18)), float(getattr(config, "MAX_ASPECT_RATIO", 0.95))
        self._lock = threading.Lock()
        self._stream = torch.cuda.Stream(device=self.device) if self.use_gpu else None
        self.net = self._init_yolo_detector()
        self._init_reid_backbone()
        self._slots: dict[int, GraphSlot] = {}
        logging.info(f"[BatchInferenceEngine] Inicializado em {self.device} (USE_GPU={self.use_gpu}, CUDAGraphs={self.use_cuda_graphs})")

    def _init_yolo_detector(self) -> torch.nn.Module:
        yolo = YOLO(self.config.YOLO_MODEL_PATH, task="detect", verbose=False)
        dummy = torch.zeros((1, 3, self.height, self.width), device=self.device, dtype=self._dtype)
        yolo.predict(dummy, device=self.device, half=self.use_gpu, verbose=False)
        net = yolo.model.to(self.device, dtype=self._dtype).eval()
        head = net.model[-1]
        head.postprocess = lambda preds: self._person_only_postprocess(preds, head.nc, self.max_dets)
        return net

    @staticmethod
    def _person_only_postprocess(preds: torch.Tensor, nc: int, max_dets: int) -> torch.Tensor:
        boxes, scores = preds.split([4, nc], dim=-1)
        person_scores = scores[..., 0:1]
        other_max = scores[..., 1:].amax(dim=-1, keepdim=True)
        furniture_max = scores[..., 56:63].amax(dim=-1, keepdim=True)
        valid = (person_scores >= other_max * 1.6) & ((furniture_max <= 0.08) | (person_scores >= 0.75))
        person_scores = torch.where(valid, person_scores, torch.zeros_like(person_scores))
        top_scores, idx = person_scores.topk(max_dets, dim=1)
        top_boxes = boxes.gather(dim=1, index=idx.expand(-1, -1, 4))
        return torch.cat([top_boxes, top_scores, torch.zeros_like(top_scores)], dim=-1)

    def _init_reid_backbone(self) -> None:
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features.to(self.device, dtype=self._dtype).eval()
        self.backbone_mid, self.backbone_deep = self.backbone[:9], self.backbone[9:]
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self._dtype).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self._dtype).view(1, 3, 1, 1)

    def _run_forward_pass(self, slot: GraphSlot) -> tuple[torch.Tensor, torch.Tensor]:
        t_rgb = slot.static_bgr.flip(-1).permute(0, 3, 1, 2).contiguous().to(self._dtype).div_(255.0)
        preds = self.net(t_rgb)[0]
        b_x1, b_y1 = preds[..., 0:1].clamp(0.0, float(self.width)), preds[..., 1:2].clamp(0.0, float(self.height))
        b_x2, b_y2 = preds[..., 2:3].clamp(0.0, float(self.width)), preds[..., 3:4].clamp(0.0, float(self.height))
        clamped = torch.cat([b_x1, b_y1, b_x2, b_y2], dim=-1).to(self._dtype)
        rois = torch.cat([slot.batch_idx_static, clamped], dim=-1).reshape(-1, 5)
        crops = roi_align(t_rgb, rois, output_size=(128, 64), spatial_scale=1.0, aligned=True)
        crops_norm = (crops - self._mean) / self._std
        mid_map = self.backbone_mid(crops_norm)
        feats = extract_multistripe_embeddings_from_maps(crops, mid_map, self.backbone_deep(mid_map)).view(slot.batch_size, self.max_dets, EMBEDDING_DIM)
        return preds.float(), feats.float()

    def _capture_cuda_graph(self, slot: GraphSlot) -> None:
        torch.cuda.synchronize()
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream), torch.inference_mode():
            for _ in range(4):
                slot.out_dets, slot.out_feats = self._run_forward_pass(slot)
        torch.cuda.current_stream().wait_stream(self._stream)
        slot.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(slot.graph, stream=self._stream), torch.inference_mode():
            slot.out_dets, slot.out_feats = self._run_forward_pass(slot)
        torch.cuda.synchronize()

    def _get_slot(self, batch_size: int) -> GraphSlot:
        slot = self._slots.get(batch_size)
        if slot is not None:
            return slot
        slot = GraphSlot(batch_size, self.height, self.width, self.max_dets, EMBEDDING_DIM, self.device, self._dtype)
        if self.use_cuda_graphs:
            self._capture_cuda_graph(slot)
        else:
            slot.eager_fn = lambda: self._run_forward_pass(slot)
        self._slots[batch_size] = slot
        return slot

    def _execute_slot_on_device(self, slot: GraphSlot, p_in: torch.Tensor, buf_idx: int) -> None:
        if self.use_cuda_graphs:
            with torch.cuda.stream(self._stream):
                slot.static_bgr.copy_(p_in, non_blocking=True)
                slot.graph.replay()
                slot.pinned_dets[buf_idx].copy_(slot.out_dets, non_blocking=True)
                slot.pinned_feats[buf_idx].copy_(slot.out_feats, non_blocking=True)
        elif self.use_gpu:
            with torch.cuda.stream(self._stream):
                slot.static_bgr.copy_(p_in, non_blocking=True)
                dets, feats = slot.eager_fn()
                slot.pinned_dets[buf_idx].copy_(dets, non_blocking=True)
                slot.pinned_feats[buf_idx].copy_(feats, non_blocking=True)
        else:
            slot.static_bgr.copy_(p_in)
            dets, feats = slot.eager_fn()
            slot.pinned_dets[buf_idx].copy_(dets)
            slot.pinned_feats[buf_idx].copy_(feats)

    @torch.inference_mode()
    def launch_async(self, frames: list[np.ndarray], buf_idx: int = 0) -> tuple[GraphSlot, int]:
        slot = self._get_slot(len(frames))
        p_in = slot.pinned_in[buf_idx]
        for i, frame in enumerate(frames):
            p_in[i].copy_(torch.from_numpy(frame))
        self._execute_slot_on_device(slot, p_in, buf_idx)
        return slot, buf_idx

    def _filter_single_camera(self, c_dets: np.ndarray, c_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1, y1 = np.clip(c_dets[:, 0], 0.0, self.width), np.clip(c_dets[:, 1], 0.0, self.height)
        x2, y2 = np.clip(c_dets[:, 2], 0.0, self.width), np.clip(c_dets[:, 3], 0.0, self.height)
        confs, w, h = c_dets[:, 4], x2 - x1, y2 - y1
        ar, area = w / np.maximum(1.0, h), w * h
        sliver = ((x1 <= 3.0) | (x2 >= self.width - 3.0)) & (w < 22.0) & (confs < 0.58)
        valid = (confs >= self.conf_low) & (w >= self.min_w) & (h >= self.min_h) & (ar >= self.min_ar) & (ar <= self.max_ar) & (area <= float(self.width * self.height) * 0.60) & (~sliver)
        if not np.any(valid):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        v_boxes = np.column_stack((x1[valid], y1[valid], x2[valid], y2[valid])).astype(np.float32)
        v_confs, v_feats = confs[valid].astype(np.float32), c_feats[valid].copy()
        if len(v_boxes) > 1:
            keep = self._filter_boxes_torchvision(v_boxes, v_confs)
            v_boxes, v_confs, v_feats = v_boxes[keep], v_confs[keep], v_feats[keep]
        return v_boxes, v_confs, v_feats

    def synchronize_and_filter(self, slot: GraphSlot, buf_idx: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._stream is not None:
            self._stream.synchronize()
        dets_b = slot.pinned_dets[buf_idx].numpy()
        feats_b = slot.pinned_feats[buf_idx].numpy()
        return [self._filter_single_camera(dets_b[c], feats_b[c]) for c in range(slot.batch_size)]

    def _filter_boxes_torchvision(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        boxes_t, scores_t = torch.from_numpy(boxes), torch.from_numpy(scores)
        keep_idx = nms(boxes_t, scores_t, self.iou_thresh)
        if len(keep_idx) <= 1:
            return keep_idx.numpy()
        b, s = boxes_t[keep_idx], scores_t[keep_idx]
        w, h = (b[:, 2] - b[:, 0]).clamp(min=1.0), (b[:, 3] - b[:, 1]).clamp(min=1.0)
        cx, areas = 0.5 * (b[:, 0] + b[:, 2]), w * h
        inter_wh = (torch.minimum(b[:, None, 2:], b[None, :, 2:]) - torch.maximum(b[:, None, :2], b[None, :, :2])).clamp(min=0.0)
        io_min = (inter_wh[:, :, 0] * inter_wh[:, :, 1]) / torch.minimum(areas[:, None], areas[None, :])
        iou = box_iou(b, b)
        lower = torch.triu(torch.ones((len(b), len(b)), dtype=torch.bool), diagonal=1)
        dx = torch.abs(cx[:, None] - cx[None, :]) / torch.maximum(w[:, None], w[None, :])
        scale = torch.minimum(h[:, None], h[None, :]) / torch.maximum(h[:, None], h[None, :])
        suppressed = ((lower & (iou > 0.44) & (dx < 0.15) & (scale > 0.78)) | (lower & (io_min > 0.84) & (scale < 0.72) & (dx < 0.20) & (s[None, :] < 0.55))).any(dim=0)
        return keep_idx[~suppressed].numpy()

    def infer_batch(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        with self._lock:
            slot, buf_idx = self.launch_async(frames, buf_idx=0)
            return self.synchronize_and_filter(slot, buf_idx)

    def cleanup(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._slots.clear()
