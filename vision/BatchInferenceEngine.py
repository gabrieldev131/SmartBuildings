# vision/BatchInferenceEngine.py
"""
Batched Inference Engine with GPU (FP16 + CUDA Graphs) and CPU (FP32) support.
Uses Ultralytics YOLO26n (`one2one` head with non-person/furniture class suppression)
and Torchvision (`roi_align`, `nms`, `box_iou`, `mobilenet_v3_small`) for Re-ID extraction.
"""
import logging
import threading
import numpy as np
import torch
import torchvision.models as models
from torchvision.ops import box_iou, nms, roi_align
from ultralytics import YOLO

from vision.featureExtractor import EMBEDDING_DIM, extract_multistripe_embeddings_from_maps


class _GraphSlot:
    """Static device and pinned host buffers for a specific camera batch size B."""

    def __init__(
        self,
        batch_size: int,
        height: int,
        width: int,
        max_dets: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.batch_size = batch_size
        use_pin = (device.type == "cuda") and torch.cuda.is_available()

        self.pinned_in = [
            torch.empty((batch_size, height, width, 3), dtype=torch.uint8, pin_memory=use_pin)
            for _ in range(2)
        ]
        self.pinned_dets = [
            torch.empty((batch_size, max_dets, 6), dtype=torch.float32, pin_memory=use_pin)
            for _ in range(2)
        ]
        self.pinned_feats = [
            torch.empty((batch_size, max_dets, EMBEDDING_DIM), dtype=torch.float32, pin_memory=use_pin)
            for _ in range(2)
        ]

        self.static_bgr = torch.empty((batch_size, height, width, 3), device=device, dtype=torch.uint8)
        self.batch_idx_static = (
            torch.arange(batch_size, device=device, dtype=dtype)
            .view(batch_size, 1, 1)
            .expand(batch_size, max_dets, 1)
        )
        self.out_dets: torch.Tensor = None
        self.out_feats: torch.Tensor = None
        self.graph: torch.cuda.CUDAGraph = None
        self._eager_fn = None


class BatchInferenceEngine:
    """Centralized multi-camera person detection and Re-ID feature extraction engine."""

    def __init__(self, config):
        self.config = config
        self.width = getattr(config, "PROCESSING_WIDTH", 640)
        self.height = getattr(config, "PROCESSING_HEIGHT", 480)
        self.max_dets = getattr(config, "MAX_DETECTIONS_PER_CAMERA", 18)

        self.use_gpu = bool(getattr(config, "USE_GPU", True)) and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        self.use_cuda_graphs = bool(getattr(config, "USE_CUDA_GRAPHS", True)) and self.use_gpu
        self._dtype = torch.float16 if self.use_gpu else torch.float32

        self.conf_low = getattr(config, "YOLO_CONF_LOW", 0.35)
        self.iou_thresh = getattr(config, "YOLO_IOU_THRESHOLD", 0.50)

        self.min_w = float(getattr(config, "MIN_BOX_WIDTH", 16))
        self.min_h = float(getattr(config, "MIN_BOX_HEIGHT", 34))
        self.min_ar = float(getattr(config, "MIN_ASPECT_RATIO", 0.18))
        self.max_ar = float(getattr(config, "MAX_ASPECT_RATIO", 0.95))

        self._lock = threading.Lock()
        self._stream = torch.cuda.Stream(device=self.device) if self.use_gpu else None

        # Load and fuse YOLO26n model (uses the clean one2one inference head)
        yolo = YOLO(self.config.YOLO_MODEL_PATH, task="detect", verbose=False)
        dummy = torch.zeros((1, 3, self.height, self.width), device=self.device, dtype=self._dtype)
        yolo.predict(dummy, device=self.device, half=self.use_gpu, verbose=False)
        self.net = yolo.model.to(self.device, dtype=self._dtype).eval()

        head = self.net.model[-1]
        max_dets = self.max_dets
        nc = head.nc

        def _person_only_postprocess(preds: torch.Tensor) -> torch.Tensor:
            boxes, scores = preds.split([4, nc], dim=-1)
            person_scores = scores[..., 0:1]
            other_max = scores[..., 1:].amax(dim=-1, keepdim=True)
            # Suppress anchors with chair/couch/furniture activation (COCO classes 56..62)
            furniture_max = scores[..., 56:63].amax(dim=-1, keepdim=True)
            is_valid_person = (person_scores >= other_max * 1.6) & (
                (furniture_max <= 0.08) | (person_scores >= 0.75)
            )
            person_scores = torch.where(is_valid_person, person_scores, torch.zeros_like(person_scores))

            top_scores, idx = person_scores.topk(max_dets, dim=1)
            top_boxes = boxes.gather(dim=1, index=idx.expand(-1, -1, 4))
            return torch.cat([top_boxes, top_scores, torch.zeros_like(top_scores)], dim=-1)

        head.postprocess = _person_only_postprocess

        # Torchvision MobileNetV3-Small Re-ID backbone
        self.backbone = (
            models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            .features.to(self.device, dtype=self._dtype)
            .eval()
        )
        self.backbone_mid = self.backbone[:9]
        self.backbone_deep = self.backbone[9:]
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self._dtype).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self._dtype).view(1, 3, 1, 1)

        self._slots: dict[int, _GraphSlot] = {}
        logging.info(
            f"[BatchInferenceEngine] Inicializado em {self.device} "
            f"(USE_GPU={self.use_gpu}, CUDAGraphs={self.use_cuda_graphs})"
        )

    def _get_slot(self, batch_size: int) -> _GraphSlot:
        slot = self._slots.get(batch_size)
        if slot is not None:
            return slot

        slot = _GraphSlot(batch_size, self.height, self.width, self.max_dets, self.device, self._dtype)
        dtype = self._dtype

        def _forward_pass():
            t_rgb = slot.static_bgr.flip(-1).permute(0, 3, 1, 2).contiguous().to(dtype).div_(255.0)
            preds = self.net(t_rgb)[0]
            clamped_boxes = torch.cat(
                [
                    preds[..., 0:1].clamp(0.0, float(self.width)),
                    preds[..., 1:2].clamp(0.0, float(self.height)),
                    preds[..., 2:3].clamp(0.0, float(self.width)),
                    preds[..., 3:4].clamp(0.0, float(self.height)),
                ],
                dim=-1,
            ).to(dtype)
            rois = torch.cat([slot.batch_idx_static, clamped_boxes], dim=-1).reshape(-1, 5)
            crops = roi_align(t_rgb, rois, output_size=(128, 64), spatial_scale=1.0, aligned=True)

            crops_norm = (crops - self._mean) / self._std
            mid_map = self.backbone_mid(crops_norm)
            deep_map = self.backbone_deep(mid_map)
            feats = extract_multistripe_embeddings_from_maps(crops, mid_map, deep_map).view(
                batch_size, self.max_dets, EMBEDDING_DIM
            )
            return preds.float(), feats.float()

        if self.use_cuda_graphs:
            torch.cuda.synchronize()
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream), torch.inference_mode():
                for _ in range(4):
                    slot.out_dets, slot.out_feats = _forward_pass()
            torch.cuda.current_stream().wait_stream(self._stream)

            slot.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(slot.graph, stream=self._stream), torch.inference_mode():
                slot.out_dets, slot.out_feats = _forward_pass()
            torch.cuda.synchronize()
        else:
            slot._eager_fn = _forward_pass

        self._slots[batch_size] = slot
        return slot

    @torch.inference_mode()
    def launch_async(self, frames: list[np.ndarray], buf_idx: int = 0) -> tuple[_GraphSlot, int]:
        slot = self._get_slot(len(frames))
        p_in = slot.pinned_in[buf_idx]
        for i, frame in enumerate(frames):
            p_in[i].copy_(torch.from_numpy(frame))

        if self.use_cuda_graphs:
            with torch.cuda.stream(self._stream):
                slot.static_bgr.copy_(p_in, non_blocking=True)
                slot.graph.replay()
                slot.pinned_dets[buf_idx].copy_(slot.out_dets, non_blocking=True)
                slot.pinned_feats[buf_idx].copy_(slot.out_feats, non_blocking=True)
        elif self.use_gpu:
            with torch.cuda.stream(self._stream):
                slot.static_bgr.copy_(p_in, non_blocking=True)
                dets, feats = slot._eager_fn()
                slot.pinned_dets[buf_idx].copy_(dets, non_blocking=True)
                slot.pinned_feats[buf_idx].copy_(feats, non_blocking=True)
        else:
            slot.static_bgr.copy_(p_in)
            dets, feats = slot._eager_fn()
            slot.pinned_dets[buf_idx].copy_(dets)
            slot.pinned_feats[buf_idx].copy_(feats)

        return slot, buf_idx

    def synchronize_and_filter(
        self, slot: _GraphSlot, buf_idx: int
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._stream is not None:
            self._stream.synchronize()

        dets_batch = slot.pinned_dets[buf_idx].numpy()
        feats_batch = slot.pinned_feats[buf_idx].numpy()
        frame_area = float(self.width * self.height)

        results_per_cam: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for c in range(slot.batch_size):
            c_dets = dets_batch[c]
            c_feats = feats_batch[c]

            x1 = np.clip(c_dets[:, 0], 0.0, self.width)
            y1 = np.clip(c_dets[:, 1], 0.0, self.height)
            x2 = np.clip(c_dets[:, 2], 0.0, self.width)
            y2 = np.clip(c_dets[:, 3], 0.0, self.height)
            confs = c_dets[:, 4]

            w, h = x2 - x1, y2 - y1
            ar = w / np.maximum(1.0, h)
            area = w * h
            edge_sliver = ((x1 <= 3.0) | (x2 >= self.width - 3.0)) & (w < 22.0) & (confs < 0.58)

            valid = (
                (confs >= self.conf_low)
                & (w >= self.min_w)
                & (h >= self.min_h)
                & (ar >= self.min_ar)
                & (ar <= self.max_ar)
                & (area <= frame_area * 0.60)
                & (~edge_sliver)
            )

            if not np.any(valid):
                results_per_cam.append((
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0, EMBEDDING_DIM), dtype=np.float32),
                ))
                continue

            v_boxes = np.column_stack((x1[valid], y1[valid], x2[valid], y2[valid])).astype(np.float32)
            v_confs = confs[valid].astype(np.float32)
            v_feats = c_feats[valid].copy()

            if len(v_boxes) > 1:
                keep = self._filter_boxes_torchvision(v_boxes, v_confs)
                v_boxes, v_confs, v_feats = v_boxes[keep], v_confs[keep], v_feats[keep]

            results_per_cam.append((v_boxes, v_confs, v_feats))

        return results_per_cam

    def _filter_boxes_torchvision(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply Torchvision C++ `nms` and suppress contained partial-body sub-boxes."""
        boxes_t = torch.from_numpy(boxes)
        scores_t = torch.from_numpy(scores)
        keep_idx = nms(boxes_t, scores_t, self.iou_thresh)
        if len(keep_idx) <= 1:
            return keep_idx.numpy()

        b = boxes_t[keep_idx]
        s = scores_t[keep_idx]
        w = (b[:, 2] - b[:, 0]).clamp(min=1.0)
        h = (b[:, 3] - b[:, 1]).clamp(min=1.0)
        cx = 0.5 * (b[:, 0] + b[:, 2])
        areas = w * h

        inter_tl = torch.maximum(b[:, None, :2], b[None, :, :2])
        inter_br = torch.minimum(b[:, None, 2:], b[None, :, 2:])
        inter_wh = (inter_br - inter_tl).clamp(min=0.0)
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]
        io_min = inter_area / torch.minimum(areas[:, None], areas[None, :])
        iou = box_iou(b, b)

        lower_rank = torch.triu(torch.ones((len(b), len(b)), dtype=torch.bool), diagonal=1)
        dx_norm = torch.abs(cx[:, None] - cx[None, :]) / torch.maximum(w[:, None], w[None, :])
        scale_ratio = torch.minimum(h[:, None], h[None, :]) / torch.maximum(h[:, None], h[None, :])

        same_center_dup = lower_rank & (iou > 0.44) & (dx_norm < 0.15) & (scale_ratio > 0.78)
        subpart_dup = lower_rank & (io_min > 0.84) & (scale_ratio < 0.72) & (dx_norm < 0.20) & (s[None, :] < 0.55)
        suppressed = (same_center_dup | subpart_dup).any(dim=0)
        return keep_idx[~suppressed].numpy()

    def infer_batch(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        with self._lock:
            slot, buf_idx = self.launch_async(frames, buf_idx=0)
            return self.synchronize_and_filter(slot, buf_idx)

    def cleanup(self) -> None:
        try:
            if self._stream is not None:
                self._stream.synchronize()
            self._slots.clear()
        except Exception:
            pass
