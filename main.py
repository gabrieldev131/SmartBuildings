# main.py
import config
import logging
from multiprocessing import freeze_support
from model.NetworkScanner import NetworkScanner
from model.FrameReader import FrameReader 
from view.console_view import ConsoleView
from controller.AppController import AppController

def setup_logging():
    """Configura o sistema de logs para monitoramento via console/Docker."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def main():
    # 1. Configuração Inicial
    setup_logging()
    view = ConsoleView()
    
    scanner = NetworkScanner(
        config.NETWORK_BASE, config.IP_RANGE_TO_SCAN, config.RTSP_PORT
    )

    model = {
        'scanner': scanner,
        'frame_reader_class': FrameReader # Passamos a classe FrameReader
    }

    controller = AppController(model, view, config)
    controller.run()

if __name__ == "__main__":
    # ESSENCIAL: Garante que o Multiprocessing funcione em Windows e ambientes Docker
    freeze_support()
    main()

