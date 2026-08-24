# extraction/frameReaderCommand/ReadRTSPCommand.py
import cv2
import logging
import time
from typing import Tuple, Optional, Any
from extraction.frameReaderCommand.IFrameCommand import IFrameCommand
import os

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
        
        # Limita buffers e reduz a latência da CPU.
        # "threads;1" limita as threads internas de decodificação do FFMPEG:
        # sem isso, o próprio decoder pode criar várias threads (uma por
        # núcleo) só para decodificar o vídeo, mais uma fonte de threads
        # nativas competindo pela CPU com o YOLO/OpenCV/BLAS.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|threads;1"
        
        # Força explicitamente a utilização do backend FFMPEG
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        
        # Diz ao OpenCV para não guardar histórico de frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
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