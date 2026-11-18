# identity/runtime_setup.py
import os

_CPU_THREADS = "2"
os.environ.setdefault("OMP_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _CPU_THREADS)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def _configure_torch_threads(num_threads: int, gpu_active: bool) -> None:
    import torch
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    torch.backends.cudnn.enabled = gpu_active
    torch.backends.cudnn.benchmark = gpu_active
    torch.backends.cuda.matmul.allow_tf32 = gpu_active
    torch.backends.cudnn.allow_tf32 = gpu_active


def apply(use_gpu: bool = None) -> None:
    """Apply OpenCV and PyTorch runtime optimizations for the current process."""
    import cv2
    import torch
    if use_gpu is None:
        from Config import Config
        use_gpu = bool(getattr(Config, "USE_GPU", True))
    gpu_active = bool(use_gpu) and torch.cuda.is_available()
    num_threads = 2 if gpu_active else max(2, min(4, os.cpu_count() or 2))
    cv2.setNumThreads(num_threads)
    cv2.setUseOptimized(True)
    _configure_torch_threads(num_threads, gpu_active)
