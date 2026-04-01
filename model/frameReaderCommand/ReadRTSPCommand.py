# model/commands/ReadRTSPCommand.py
import cv2
import logging
import time
from queue import Queue, Full
from model.frameReaderCommand.IFrameCommand import IFrameCommand

class ReadRTSPCommand(IFrameCommand):
    """
    Comando responsável por ler frames de uma stream RTSP ou arquivo de vídeo.
    """
    def __init__(self, source: str, output_queue: Queue, width: int, height: int, reconnect_delay: int = 5):
        self.source = source
        self.output_queue = output_queue
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.cap = None

    def _connect(self):
        logging.info(f"Tentando conectar a {self.source}...")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logging.error(f"Erro ao abrir fonte de vídeo: {self.source}")
            self.cap = None

    def execute(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            if self.cap is None:
                time.sleep(self.reconnect_delay)
                return

        success, frame = self.cap.read()
        if not success:
            logging.warning(f"Falha na leitura de frame em {self.source}. Tentando reconectar...")
            self.cleanup()
            return

        resized_frame = cv2.resize(frame, (self.width, self.height))
        try:
            self.output_queue.put((self.source, resized_frame), timeout=1)
        except Full:
            logging.debug(f"Fila cheia para {self.source}. Frame descartado.")

    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None