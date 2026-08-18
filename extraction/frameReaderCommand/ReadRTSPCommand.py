# extraction/frameReaderCommand/ReadRTSPCommand.py
import cv2
import logging
import time
from typing import Tuple, Optional, Any
from extraction.frameReaderCommand.IFrameCommand import IFrameCommand

class ReadRTSPCommand(IFrameCommand):
    """
    Comando responsável por ler frames de uma stream RTSP ou arquivo de vídeo.
    """
    def __init__(self, source: str, width: int, height: int, reconnect_delay: int = 5):
        self.source = source
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

    def execute(self) -> Optional[Tuple[str, Any]]:
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            if self.cap is None:
                time.sleep(self.reconnect_delay)
                return None

        success, frame = self.cap.read()
        if not success:
            logging.warning(f"Falha na leitura de frame em {self.source}. Tentando reconectar...")
            self.cleanup()
            return None

        resized_frame = cv2.resize(frame, (self.width, self.height))
        return (self.source, resized_frame)

    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None