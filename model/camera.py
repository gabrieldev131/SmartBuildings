# model/camera.py
import cv2
import time
import threading
from queue import Queue, Full # Importamos a exceção 'Full'

class FrameReader:
    """
    Lê frames de uma fonte de vídeo em uma thread dedicada.
    Gerencia reconexões automáticas e coloca frames redimensionados em uma fila
    thread-safe, evitando travamentos se o consumidor estiver lento.
    """
    def __init__(self, source: str, output_queue: Queue, stop_event: threading.Event, width: int, height: int, reconnect_delay: int = 5):
        self.source = source
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.name = f"FrameReader_{source}"

    def _read_loop(self):
        """O loop principal que roda na thread, lendo e enfileirando frames."""
        cap = None
        while not self.stop_event.is_set():
            # Se não estiver conectado, tenta (re)conectar
            if cap is None or not cap.isOpened():
                print(f"[{self.source}] Fonte de vídeo desconectada. Tentando reconectar em {self.reconnect_delay}s...")
                cap = cv2.VideoCapture(self.source)
                time.sleep(self.reconnect_delay)
                continue

            ret, frame = cap.read()

            # Se a leitura falhar, libera o objeto para forçar a reconexão no próximo ciclo
            if not ret:
                print(f"[{self.source}] Não foi possível ler o frame. A conexão pode ter sido perdida.")
                cap.release()
                continue
            
            # Se a leitura for bem-sucedida, processa e enfileira o frame
            resized_frame = cv2.resize(frame, (self.width, self.height))
            try:
                # Tenta colocar na fila com um timeout de 1 segundo
                self.output_queue.put((self.source, resized_frame), timeout=1)
            except Full:
                # Se a fila estiver cheia, o consumidor (loop principal) está muito lento.
                # Em vez de travar, nós simplesmente descartamos este frame.
                print(f"[{self.source}] AVISO: Fila de processamento cheia. Descartando frame para manter o tempo real.")
                pass
        
        if cap:
            cap.release()
        print(f"[{self.source}] Thread de leitura finalizada.")

    def start(self):
        """Inicia a thread de leitura de frames."""
        print(f"[{self.source}] Iniciando thread de leitura...")
        self.thread.start()