# main.py
#
# IMPORTANTE: as variáveis de ambiente abaixo têm que ser definidas ANTES de
# importar cv2, torch, numpy ou qualquer módulo que os importe transitivamente
# (Config, FrameReaderManagement, etc). Essas bibliotecas leem essas variáveis
# uma única vez, na inicialização, para decidir quantas threads nativas usar.
#
# Por que isso importa: o YOLO já roda na GPU (device=0). PyTorch, OpenCV e o
# backend de álgebra linear do NumPy (OpenBLAS/MKL) cada um, por padrão, cria
# seu PRÓPRIO pool de threads do tamanho do número de núcleos da CPU, mesmo
# fazendo só pré/pós-processamento leve. Isso gera oversubscription: 3 pools
# de N threads cada, competindo pelos mesmos N núcleos = mais troca de
# contexto do que trabalho útil, CPU sempre no talo, aquecimento sustentado
# e queda de FPS ao longo do tempo (exatamente o sintoma relatado).
import os

_CPU_THREADS = "2"  # ajuste conforme os núcleos livres da sua máquina
os.environ["OMP_NUM_THREADS"] = _CPU_THREADS
os.environ["MKL_NUM_THREADS"] = _CPU_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _CPU_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _CPU_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = _CPU_THREADS

import cv2
# GPU faz o trabalho pesado; a CPU só orquestra. 1-2 threads é suficiente
# para resize/cvtColor/calcHist/decodificação e evita o OpenCV competir
# pelos mesmos núcleos que o PyTorch.
cv2.setNumThreads(2)

import torch
# Threads de CPU do PyTorch (usadas em pré/pós-processamento do YOLO,
# NMS, etc). Não precisa ser igual ao nº de núcleos: o gargalo é a GPU.
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
# cudnn.benchmark: setado UMA vez para todo o processo (antes era setado
# dentro do __init__ de cada CameraWorker, redundante se houver >1 câmara).
torch.backends.cudnn.benchmark = True

import logging
from Config import Config
from extraction.FrameReaderManagement import FrameReaderManagement


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    config = Config()

    app = FrameReaderManagement(config)
    app.run()


if __name__ == "__main__":
    main()