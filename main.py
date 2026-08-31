# main.py
import multiprocessing as mp
import logging

from Config import Config
from core.runtime_setup import apply as apply_runtime_setup
from extraction.FrameReaderManagement import FrameReaderManagement


def main():
    # Configuração de logs global
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Limites de threads nativas (torch/cv2/BLAS) para ESTE processo
    # (o processo principal). Cada processo de câmera reaplica isso
    # sozinho ao iniciar (ver CameraProcess.py).
    apply_runtime_setup()

    config = Config()

    app = FrameReaderManagement(config)
    app.run()


if __name__ == "__main__":
    # OBRIGATÓRIO com CUDA: "fork" (padrão no Linux) duplicaria qualquer
    # contexto CUDA já inicializado, corrompendo-o no processo filho.
    # "spawn" inicia cada processo do zero -- necessário já que cada
    # processo de câmera cria seu próprio modelo YOLO na GPU.
    mp.set_start_method("spawn", force=True)
    main()