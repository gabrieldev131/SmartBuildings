# model/invokers/FrameReaderInvoker.py
import threading
import logging
from model.frameReaderCommand.IFrameCommand import IFrameCommand

class FrameReaderInvoker(threading.Thread):
    """
    Invoker que executa continuamente um IFrameCommand em uma thread dedicada.
    Isola a lógica de concorrência da lógica de aquisição de imagem.
    """
    def __init__(self, command: IFrameCommand, stop_event: threading.Event, name: str = "FrameReaderInvoker"):
        super().__init__(daemon=True, name=name)
        self.command = command
        self.stop_event = stop_event

    def run(self):
        logging.info(f"[{self.name}] Iniciando loop de captura...")
        
        while not self.stop_event.is_set():
            self.command.execute()
            
        logging.info(f"[{self.name}] Thread encerrada. Executando limpeza...")
        self.command.cleanup()