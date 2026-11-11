# vision/BatchInferenceEngine.py
"""
Unified Batched Inference Engine with GPU (FP16 + CUDA Graphs) and CPU (FP32) Support.
Executes multi-camera YOLO26n detection + Intra-Class NMS + ROI-Align + MobileNetV3
Multi-Stripe Mid/Deep & 12-Channel Opponent/Luminance Re-ID extraction.
Configurable via Config.USE_GPU (True = CUDA GPU if available, False = CPU).
"""
import logging
import threading
import numpy as np
import torch
import torchvision.models as models
from torchvision.ops import roi_align
from ultralytics import YOLO

from vision.featureExtractor import extract_multistripe_embeddings_from_maps


class _GraphSlot:
    """Encapsulates static device & host buffers for a specific camera batch size B."""

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
        self.height = height
        self.width = width
        self.max_dets = max_dets
        self.device = device
        self.dtype = dtype
        use_pin = (device.type == "cuda") and torch.cuda.is_available()

        # Double-buffered host input/output buffers (slot 0 and slot 1)
        self.pinned_in = [
            torch.empty((batch_size, height, width, 3), dtype=torch.uint8, pin_memory=use_pin)
            for _ in range(2)
        ]
        self.pinned_dets = [
            torch.empty((batch_size, max_dets, 6), dtype=torch.float32, pin_memory=use_pin)
            for _ in range(2)
        ]
        self.pinned_feats = [
            torch.empty((batch_size, max_dets, 576), dtype=torch.float32, pin_memory=use_pin)
            for _ in range(2)
        ]

        # Static device tensors bound to the captured CUDA Graph (or eager execution)
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
    """
    Centralized Engine for multi-camera person detection and multi-stripe Re-ID extraction.
    Supports both GPU (CUDA FP16 + CUDA Graphs) and CPU (FP32) controlled by Config.USE_GPU.
    """

    def __init__(self, config):
        self.config = config
        self.width = getattr(config, "PROCESSING_WIDTH", 640)
        self.height = getattr(config, "PROCESSING_HEIGHT", 480)
        self.max_dets = getattr(config, "MAX_DETECTIONS_PER_CAMERA", 16)

        self.use_gpu = bool(getattr(config, "USE_GPU", True)) and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        self.use_cuda_graphs = bool(getattr(config, "USE_CUDA_GRAPHS", True)) and self.use_gpu
        self._dtype = torch.float16 if self.use_gpu else torch.float32

        self.conf_low = getattr(config, "YOLO_CONF_LOW", 0.28)
        self.iou_thresh = getattr(config, "YOLO_IOU_THRESHOLD", 0.44)

        self.min_w = float(getattr(config, "MIN_BOX_WIDTH", 16))
        self.min_h = float(getattr(config, "MIN_BOX_HEIGHT", 32))
        self.min_ar = float(getattr(config, "MIN_ASPECT_RATIO", 0.15))
        self.max_ar = float(getattr(config, "MAX_ASPECT_RATIO", 2.80))

        self._lock = threading.Lock()
        self._stream = torch.cuda.Stream(device=self.device) if self.use_gpu else None

        # Load and fuse YOLO model on the selected device
        yolo = YOLO(self.config.YOLO_MODEL_PATH, task="detect", verbose=False)
        if self.use_gpu:
            dummy = torch.zeros((1, 3, self.height, self.width), device=self.device, dtype=torch.float16)
            yolo.predict(dummy, device=self.device, half=True, verbose=False)
            self.net = yolo.model.to(self.device).eval().half()
        else:
            dummy = torch.zeros((1, 3, self.height, self.width), device=self.device, dtype=torch.float32)
            yolo.predict(dummy, device="cpu", half=False, verbose=False)
            self.net = yolo.model.to(self.device).eval().float()

        # Patch Detect head postprocess to directly extract top-K class-0 (person) candidates
        head = self.net.model[-1]
        max_dets = self.max_dets
        nc = head.nc

        def _person_only_postprocess(preds: torch.Tensor) -> torch.Tensor:
            boxes, scores = preds.split([4, nc], dim=-1)
            person_scores = scores[..., 0:1]
            top_scores, idx = person_scores.topk(max_dets, dim=1)
            top_boxes = boxes.gather(dim=1, index=idx.expand(-1, -1, 4))
            cls_zero = torch.zeros_like(top_scores)
            return torch.cat([top_boxes, top_scores, cls_zero], dim=-1)

        head.postprocess = _person_only_postprocess

        # Load MobileNetV3-Small Re-ID backbone split into mid-level (layer 8) and deep (layer 12)
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        ).features.to(self.device).eval()
        if self.use_gpu:
            self.backbone.half()
        else:
            self.backbone.float()

        self.backbone_mid = self.backbone[:9]
        self.backbone_deep = self.backbone[9:]

        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self._dtype).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self._dtype).view(1, 3, 1, 1)

        self._slots: dict[int, _GraphSlot] = {}
        logging.info(
            f"[BatchInferenceEngine] Inicializado em {self.device} "
            f"(USE_GPU={self.use_gpu}, FP16={self.use_gpu}, CUDAGraphs={self.use_cuda_graphs})"
        )

    def _build_slot(self, batch_size: int) -> _GraphSlot:
        slot = _GraphSlot(batch_size, self.height, self.width, self.max_dets, self.device, self._dtype)
        dtype = self._dtype

        def _forward_pass():
            t_rgb = slot.static_bgr.flip(-1).permute(0, 3, 1, 2).contiguous().to(dtype).div_(255.0)
            preds = self.net(t_rgb)[0]  # (B, max_dets, 6)
            rois = torch.cat([slot.batch_idx_static, preds[..., :4].to(dtype)], dim=-1).reshape(-1, 5)
            crops = roi_align(t_rgb, rois, output_size=(128, 64), spatial_scale=1.0, aligned=True)

            crops_norm = (crops - self._mean) / self._std
            mid_map = self.backbone_mid(crops_norm)
            deep_map = self.backbone_deep(mid_map)
            feats = extract_multistripe_embeddings_from_maps(crops, mid_map, deep_map).view(
                batch_size, self.max_dets, 576
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
            logging.info(f"[BatchInferenceEngine] CUDA Graph capturado para batch_size={batch_size} câmaras.")
        else:
            slot._eager_fn = _forward_pass

        self._slots[batch_size] = slot
        return slot

    def _get_slot(self, batch_size: int) -> _GraphSlot:
        slot = self._slots.get(batch_size)
        if slot is None:
            slot = self._build_slot(batch_size)
        return slot

    @torch.inference_mode()
    def launch_async(self, frames: list[np.ndarray], buf_idx: int = 0) -> tuple[_GraphSlot, int]:
        """
        Copy frames into host buffer and launch inference on GPU (CUDA Graph or stream) or CPU.
        """
        b = len(frames)
        slot = self._get_slot(b)
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
            # CPU execution path
            slot.static_bgr.copy_(p_in)
            dets, feats = slot._eager_fn()
            slot.pinned_dets[buf_idx].copy_(dets)
            slot.pinned_feats[buf_idx].copy_(feats)

        return slot, buf_idx

    def synchronize_and_filter(
        self, slot: _GraphSlot, buf_idx: int
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Wait for the GPU stream (if active) to complete and return filtered
        (boxes_xyxy, confs, feats) per camera in the batch.
        Applies geometric human box validation and scale-aware intra-class NMS.
        """
        if self._stream is not None:
            self._stream.synchronize()

        dets_batch = slot.pinned_dets[buf_idx].numpy()
        feats_batch = slot.pinned_feats[buf_idx].numpy()
        frame_area = float(self.width * self.height)
        max_area = frame_area * 0.65

        results_per_cam: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for c in range(slot.batch_size):
            c_dets = dets_batch[c]
            c_feats = feats_batch[c]

            x1 = np.clip(c_dets[:, 0], 0.0, self.width)
            y1 = np.clip(c_dets[:, 1], 0.0, self.height)
            x2 = np.clip(c_dets[:, 2], 0.0, self.width)
            y2 = np.clip(c_dets[:, 3], 0.0, self.height)
            confs = c_dets[:, 4]

            w = x2 - x1
            h = y2 - y1
            ar = w / np.maximum(1.0, h)
            area = w * h
            lens_occlusion = (h >= self.height * 0.96) & (area >= frame_area * 0.45)

            # Reject narrow edge slivers (e.g. half an arm/shoulder cut off by image boundary)
            edge_sliver = ((x1 <= 3.0) | (x2 >= self.width - 3.0)) & (w < 26.0) & (confs < 0.65)
            top_bottom_artifact = ((y1 <= 2.0) & (y2 < self.height * 0.22) & (confs < 0.48))

            valid = (
                (confs >= self.conf_low)
                & (w >= self.min_w)
                & (h >= self.min_h)
                & (ar >= self.min_ar)
                & (ar <= self.max_ar)
                & (area <= max_area)
                & (~lens_occlusion)
                & (~edge_sliver)
                & (~top_bottom_artifact)
            )

            if not np.any(valid):
                results_per_cam.append((
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0, 576), dtype=np.float32),
                ))
                continue

            v_boxes = np.column_stack((x1[valid], y1[valid], x2[valid], y2[valid])).astype(np.float32)
            v_confs = confs[valid].astype(np.float32)
            v_feats = c_feats[valid].copy()

            # Intra-class NMS with part-box suppression
            if len(v_boxes) > 1:
                keep = self._nms_cpu(v_boxes, v_confs, self.iou_thresh)
                v_boxes = v_boxes[keep]
                v_confs = v_confs[keep]
                v_feats = v_feats[keep]

            results_per_cam.append((v_boxes, v_confs, v_feats))

        return results_per_cam

    @staticmethod
    def _nms_cpu(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """
        Scale-aware greedy NMS for person detections:
        - Suppresses duplicate boxes on the same person with IoU > iou_threshold.
        - Suppresses partial-body sub-boxes (e.g. head/torso box inside full-body box) via io_min
          and horizontal center alignment, while preserving high-confidence background people
          partially occluded by a larger foreground person.
        """
        order = scores.argsort()[::-1]
        keep = []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w_all = np.maximum(1.0, x2 - x1)
        h_all = np.maximum(1.0, y2 - y1)
        cx_all = x1 + 0.5 * w_all
        areas = w_all * h_all

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter)
            io_min = inter / np.minimum(areas[i], areas[rest])

            # Check if smaller box is horizontally aligned with the larger box (sub-part of same person)
            dx_norm = np.abs(cx_all[i] - cx_all[rest]) / np.maximum(w_all[i], w_all[rest])
            scale_ratio = np.minimum(h_all[i], h_all[rest]) / np.maximum(h_all[i], h_all[rest])

            # Suppress if:
            # 1) Standard IoU exceeds threshold, OR
            # 2) Strong containment (io_min > 0.72) when candidate is not a distinct high-conf small person, OR
            # 3) Moderate containment (io_min > 0.55) and horizontally aligned on the same body column
            is_subpart = (
                (io_min > 0.78)
                | ((io_min > 0.56) & (dx_norm < 0.28) & ((scores[rest] < 0.72) | (scale_ratio > 0.45)))
            )
            suppress = (iou > iou_threshold) | is_subpart
            order = rest[~suppress]
        return keep

    def infer_batch(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Synchronous convenience wrapper for single-step batch inference."""
        with self._lock:
            slot, buf_idx = self.launch_async(frames, buf_idx=0)
            return self.synchronize_and_filter(slot, buf_idx)

    def cleanup(self) -> None:
        """Synchronize CUDA stream and release captured CUDA Graphs cleanly before exit."""
        try:
            if self._stream is not None:
                self._stream.synchronize()
            self._slots.clear()
        except Exception:
            pass
