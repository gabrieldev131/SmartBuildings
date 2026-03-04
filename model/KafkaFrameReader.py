# model/KafkaFrameReader.py
import cv2
import numpy as np
import threading
import logging
from multiprocessing import Queue as MPQueue
from confluent_kafka import Consumer


class KafkaFrameReader:
    """
    Substitui o FrameReader (RTSP) como fonte de frames.

    Adapta o TurboKafkaReader ao contrato esperado pelo AppController:
      - Recebe uma MPQueue de saida e coloca tuplas (cam_id, frame) nela,
        exatamente como o FrameReader fazia com streams RTSP.
      - Descobre automaticamente os IDs de cameras presentes no topico
        (cada mensagem Kafka tem a chave = cam_id).
      - Expoe self.thread para que o AppController possa fazer join() no shutdown.

    Estrategia de leitura (mesma do TurboKafkaReader original):
      - consume() em batch de 50 msgs por vez para drenar o buffer rapidamente.
      - Dentro do batch, so o ULTIMO frame de cada camera e decodificado,
        descartando frames antigos que nunca seriam exibidos a tempo.
      - imdecode so e chamado para o frame vencedor, economizando CPU.
    """

    def __init__(self, output_queue: MPQueue, stop_event: threading.Event,
                 bootstrap_servers: str, topic: str, group_id: str,
                 width: int, height: int,
                 target_camera_id: str = None):
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.width = width
        self.height = height
        self.target_camera_id = target_camera_id

        conf = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "latest",
            # Buffer de rede grande para nao travar o TCP
            "socket.receive.buffer.bytes": 10 * 1024 * 1024,
            # Tamanho maximo de cada mensagem (frame JPEG comprimido)
            "fetch.message.max.bytes": 5 * 1024 * 1024,
            # Baixa latencia: nao espera o buffer encher
            "fetch.wait.max.ms": 5,
            # Sem commit de offset: video nao precisa de persistencia
            "enable.auto.commit": False,
        }

        self._consumer = Consumer(conf)
        self._consumer.subscribe([topic])

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.name = f"KafkaFrameReader[{topic}]"

    def start(self):
        logging.info(f"[KafkaFrameReader] Iniciado. Alvo: "
                     f"{self.target_camera_id if self.target_camera_id else 'TODAS as cameras'}")
        self.thread.start()

    def _run(self):
        """
        Loop principal: drena o buffer Kafka em batches e coloca
        apenas o frame mais recente de cada camera na fila de saida.
        """
        while not self.stop_event.is_set():
            # --- Etapa 1: Drenagem em batch (operacao C nativa, rapida) ---
            msgs = self._consumer.consume(num_messages=50, timeout=0.01)
            if not msgs:
                continue

            # Guarda apenas a ultima mensagem de cada camera no lote.
            # Se chegaram 10 frames da cam1, so o ultimo importa.
            batch_latest: dict[str, bytes] = {}
            for msg in msgs:
                if msg.error():
                    continue
                key_bytes = msg.key()
                if not key_bytes:
                    continue
                cam_id = key_bytes.decode("utf-8")
                if self.target_camera_id and cam_id != self.target_camera_id:
                    continue
                batch_latest[cam_id] = msg.value()

            # --- Etapa 2: Decodificacao so dos frames vencedores ---
            for cam_id, img_bytes in batch_latest.items():
                frame = self._decode_frame(img_bytes)
                if frame is None:
                    continue

                # Redimensiona para resolucao de processamento configurada
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Coloca na fila no mesmo formato que o FrameReader usava:
                # (source_id, frame) — o AppController nao precisa mudar.
                try:
                    self.output_queue.put_nowait((cam_id, frame))
                except Exception:
                    # Fila cheia: descarta o frame antigo (preferimos frescor)
                    pass

        self._consumer.close()
        logging.info("[KafkaFrameReader] Thread encerrada.")

    def _decode_frame(self, img_bytes: bytes):
        """Decodifica bytes JPEG/PNG para numpy array BGR."""
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame  # None se a decodificacao falhar
        except Exception as e:
            logging.debug(f"[KafkaFrameReader] Falha ao decodificar frame: {e}")
            return None