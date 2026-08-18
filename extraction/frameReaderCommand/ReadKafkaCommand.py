# extraction/frameReaderCommand/ReadKafkaCommand.py
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Any
from confluent_kafka import Consumer
from extraction.frameReaderCommand.IFrameCommand import IFrameCommand

class ReadKafkaCommand(IFrameCommand):
    """
    Comando responsável por drenar um tópico Kafka e extrair os frames de forma síncrona.
    """
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str, width: int, height: int, target_camera_id: str = ""):
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

    def execute(self) -> Optional[Tuple[str, Any]]:
        msgs = self._consumer.consume(num_messages=1, timeout=0.05)
        if not msgs:
            return None

        msg = msgs[0]
        if msg.error():
            return None
        
        key = msg.key()
        if not key:
            return None
            
        cam_id = key.decode("utf-8")
        if self.target_camera_id and cam_id != self.target_camera_id:
            return None

        img_bytes = msg.value()
        if img_bytes is None:
            return None

        frame = self._decode_frame(img_bytes)
        if frame is None:
            return None

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        return (cam_id, frame)

    def _decode_frame(self, img_bytes: bytes):
        try:
            if img_bytes is None:
                return None
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.debug(f"Falha ao decodificar frame: {e}")
            return None

    def cleanup(self) -> None:
        if self._consumer:
            self._consumer.close()