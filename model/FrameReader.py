import cv2
import time
import threading
import logging
from queue import Queue, Full

# Configuração básica de logging para substituir os 'prints'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(threadName)s] - %(message)s')

class FrameReader:
    """
    Gerencia a captura de frames de uma fonte de vídeo de forma assíncrona.
    
    Esta classe encapsula a lógica de conexão, redimensionamento e 
    enfileiramento de frames em uma thread dedicada para evitar bloqueios.
    """

    def __init__(self, source: str, output_queue: Queue, stop_event: threading.Event, 
                 width: int, height: int, reconnect_delay: int = 5):
        self.source = source
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        
        # Inicialização da Thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.name = f"FrameReader_{source.split('/')[-1]}" # Nome simplificado para log

    def start(self):
        """Inicia a execução da thread de leitura."""
        logging.info(f"Iniciando captura da fonte: {self.source}")
        self.thread.start()

    def _run(self):
        """Loop principal de execução da thread."""
        cap = None
        while not self.stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap = self._connect()
                if cap is None:
                    time.sleep(self.reconnect_delay)
                    continue

            success, frame = cap.read()

            if not success:
                logging.warning(f"Falha na leitura de frame em {self.source}. Tentando reconectar...")
                cap.release()
                cap = None
                continue

            self._process_and_enqueue(frame)

        # Cleanup final
        if cap:
            cap.release()
        logging.info(f"Thread de leitura finalizada para {self.source}")

    def _connect(self):
        """Tenta estabelecer conexão com a fonte de vídeo."""
        logging.info(f"Tentando conectar a {self.source}...")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logging.error(f"Erro ao abrir fonte de vídeo: {self.source}")
            return None
        return cap

    def _process_and_enqueue(self, frame):
        """Redimensiona o frame e tenta inseri-lo na fila de saída."""
        resized_frame = cv2.resize(frame, (self.width, self.height))
        
        try:
            # Enfileira tupla com (ID_da_fonte, frame)
            self.output_queue.put((self.source, resized_frame), timeout=1)
        except Full:
            logging.debug(f"Fila cheia para {self.source}. Frame descartado.")