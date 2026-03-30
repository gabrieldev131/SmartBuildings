# model/commands/ReadKafkaCommand.py
import cv2
import numpy as np
import logging
from multiprocessing import Queue as MPQueue
from confluent_kafka import Consumer
from .IFrameCommand import IFrameCommand

class ReadKafkaCommand(IFrameCommand):
    """
    Comando responsável por drenar um tópico Kafka e extrair os frames mais recentes.
    """
    def __init__(self, output_queue: MPQueue, bootstrap_servers: str, topic: str, group_id: str, width: int, height: int, target_camera_id: str = None):
        self.output_queue = output_queue
        self.width = width
        self.height = height
        self.target_camera_id = target_camera_id
        
        conf = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "latest",
            "socket.receive.buffer.bytes": 10 * 1024 * 1024,
            "fetch.message.max.bytes": 5 * 1024 * 1024,
            "fetch.wait.max.ms": 5,
            "enable.auto.commit": False,
        }
        
        self._consumer = Consumer(conf)
        self._consumer.subscribe([topic])

    def execute(self) -> None:
        # Etapa 1: Drenagem em batch
        msgs = self._consumer.consume(num_messages=50, timeout=0.01)
        if not msgs:
            return

        batch_latest = {}
        for msg in msgs:
            if msg.error() or not msg.key():
                continue
            cam_id = msg.key().decode("utf-8")
            if self.target_camera_id and cam_id != self.target_camera_id:
                continue
            batch_latest[cam_id] = msg.value()

        # Etapa 2: Decodificação
        for cam_id, img_bytes in batch_latest.items():
            frame = self._decode_frame(img_bytes)
            if frame is None:
                continue

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            try:
                self.output_queue.put_nowait((cam_id, frame))
            except Exception:
                pass # Fila cheia, descarta frame antigo

    def _decode_frame(self, img_bytes: bytes):
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.debug(f"Falha ao decodificar frame: {e}")
            return None

    def cleanup(self) -> None:
        if self._consumer:
            self._consumer.close()