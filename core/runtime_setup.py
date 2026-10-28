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


def apply():
    """Apply OpenCV and PyTorch runtime optimizations for the current process."""
    import cv2
    cv2.setNumThreads(2)
    cv2.setUseOptimized(True)

    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # Interop threads can only be set once per process before parallel work starts
        pass

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
