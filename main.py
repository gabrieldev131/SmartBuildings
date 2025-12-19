# main.py
import config
from multiprocessing import freeze_support
from model.network_scanner import NetworkScanner
from model.camera import FrameReader 
from view.console_view import ConsoleView
from controller.app_controller import AppController

if __name__ == "__main__":
    # ESSENCIAL para o multiprocessing funcionar corretamente no Windows
    freeze_support()

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