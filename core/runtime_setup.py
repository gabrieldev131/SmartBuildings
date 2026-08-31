# core/runtime_setup.py
#
# Configuração de threads nativas que precisa ser aplicada em TODO processo
# Python do sistema -- o processo principal e cada processo de câmera --
# porque cada processo tem seu próprio interpretador, seu próprio PyTorch,
# seu próprio OpenCV, cada um com seus próprios pools de threads internos.
#
# As variáveis de ambiente (OMP/MKL/OPENBLAS) precisam ser setadas ANTES de
# importar numpy/torch/cv2 nesse processo -- por isso isto fica num módulo
# separado, importado bem no topo de cada ponto de entrada (main.py e
# CameraProcess.py).
import os

_CPU_THREADS = "2"  # ajuste conforme os núcleos livres da sua máquina e o nº de câmaras
os.environ.setdefault("OMP_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _CPU_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _CPU_THREADS)


def apply():
    """Chamar uma vez, bem no início de CADA processo (main e cada câmera)."""
    import cv2
    cv2.setNumThreads(2)

    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True