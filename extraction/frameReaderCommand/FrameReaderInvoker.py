# extraction/frameReaderCommand/FrameReaderInvoker.py
import threading
import logging
from extraction.frameReaderCommand.IFrameCommand import IFrameCommand


class FrameReaderInvoker(threading.Thread):
    """
    Executa continuamente um IFrameCommand numa thread dedicada (uma por
    câmara/fonte de vídeo) e mantém disponível SOMENTE o frame mais recente
    lido, protegido por um lock leve.

    Por que "apenas o mais recente" em vez de uma fila: captura (I/O de rede
    + decodificação) e processamento (YOLO/DeepSORT na GPU) rodam em
    velocidades diferentes. Se guardássemos todos os frames numa fila, o
    processamento mais lento acumularia atraso (fila crescendo = latência
    cada vez maior). Guardando só o último frame:
      - a thread de captura NUNCA fica bloqueada esperando o processamento
        (ela sobrescreve o frame anterior se ele ainda não foi consumido);
      - o processamento sempre trabalha com a imagem mais atual da câmara;
      - o custo de sincronização é um único lock, sem fila, sem cópia extra.

    Isto é o que de fato desacopla a decodificação de vídeo (que antes
    bloqueava o loop principal a cada chamada de .execute()) da inferência
    da GPU, permitindo que as duas aconteçam em paralelo.
    """
    def __init__(self, cam_key: str, command: IFrameCommand, stop_event: threading.Event, name: str = "FrameReaderInvoker"):
        super().__init__(daemon=True, name=name)
        self.cam_key = cam_key
        self.command = command
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self._latest_frames = {}

    def run(self):
        logging.info(f"[{self.name}] Iniciando loop de captura para '{self.cam_key}'...")

        while not self.stop_event.is_set():
            result = self.command.execute()
            if result is not None:
                cam_id, frame = result
                with self._lock:
                    self._latest_frames[cam_id] = frame

        logging.info(f"[{self.name}] Thread encerrada. Executando limpeza...")
        self.command.cleanup()

    def get_latest(self):
        """
        Retorna o dicionário de frames mais recentes e o consome.
        Retorna None se nenhum novo frame estiver disponível.
        """
        with self._lock:
            if not self._latest_frames:
                return None
            
            # Copiamos o estado atual e limpamos para forçar a busca de frames novos
            result = self._latest_frames.copy()
            self._latest_frames.clear()
            return result