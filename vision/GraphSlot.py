# vision/GraphSlot.py
import torch


class GraphSlot:
    """Static device and double-buffered pinned host buffers for a specific camera batch size B (SRP)."""

    def __init__(self, batch_size: int, height: int, width: int, max_dets: int, emb_dim: int, device: torch.device, dtype: torch.dtype):
        self.batch_size = batch_size
        use_pin = (device.type == "cuda") and torch.cuda.is_available()
        self.pinned_in = [torch.empty((batch_size, height, width, 3), dtype=torch.uint8, pin_memory=use_pin) for _ in range(2)]
        self.pinned_dets = [torch.empty((batch_size, max_dets, 6), dtype=torch.float32, pin_memory=use_pin) for _ in range(2)]
        self.pinned_feats = [torch.empty((batch_size, max_dets, emb_dim), dtype=torch.float32, pin_memory=use_pin) for _ in range(2)]
        self.static_bgr = torch.empty((batch_size, height, width, 3), device=device, dtype=torch.uint8)
        self.batch_idx_static = torch.arange(batch_size, device=device, dtype=dtype).view(batch_size, 1, 1).expand(batch_size, max_dets, 1)
        self.out_dets: torch.Tensor = None
        self.out_feats: torch.Tensor = None
        self.graph: torch.cuda.CUDAGraph = None
        self.eager_fn = None
