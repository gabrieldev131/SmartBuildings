# core/runtime_setup.py
"""
Native runtime configuration applied at process startup.
Limits OpenMP/MKL/OpenBLAS/OpenCV thread pools to prevent CPU oversubscription
and enables CUDA TensorCore/cuDNN optimizations for RTX GPUs.
"""
import os

_CPU_THREADS = "2"
os.environ.setdefault("OMP_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _CPU_THREADS)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def apply(use_gpu: bool = None):
    """Apply OpenCV and PyTorch runtime optimizations for the current process."""
    if use_gpu is None:
        try:
            from Config import Config
            use_gpu = bool(getattr(Config, "USE_GPU", True))
        except Exception:
            use_gpu = True

    import cv2
    import torch

    gpu_active = bool(use_gpu) and torch.cuda.is_available()
    num_threads = 2 if gpu_active else max(2, min(4, os.cpu_count() or 2))

    cv2.setNumThreads(num_threads)
    cv2.setUseOptimized(True)

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # Interop threads can only be set once per process before parallel work starts
        pass

    if gpu_active:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cudnn.enabled = False
