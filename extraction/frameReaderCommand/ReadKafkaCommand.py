# extraction/frameReaderCommand/ReadKafkaCommand.py
from collections import deque
import logging
import os
import time
from typing import Any, Optional, Tuple
import cv2
import numpy as np
from confluent_kafka import Consumer, TopicPartition, OFFSET_END

from extraction.frameReaderCommand.IFrameCommand import IFrameCommand


class ReadKafkaCommand(IFrameCommand):
    """
    Concrete Command (Command Pattern) responsible for consuming and decoding video frames
    from an Apache Kafka topic while draining stale messages per camera key to minimize latency.

    Optimized for KRaft brokers with `cleanup.policy=compact,delete` and multi-camera keyed streams:
    - Uses direct partition assignment (`assign`) with watermark tail offsets to bypass the
      3s-45s consumer-group rebalance delay and immediately fetch the latest compacted frame per camera.
    - Groups drained messages by camera key (`CAMERA_ID`) so fast cameras never starve slower cameras
      sharing the same topic.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        width: int,
        height: int,
        target_camera_id: Optional[str] = "",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.width = width
        self.height = height
        self.target_camera_id = target_camera_id or ""
        self._pending_frames: deque[Tuple[str, np.ndarray]] = deque()
        self._assigned = False
        self._last_assign_attempt = 0.0
        self._warned_no_topic = False

        conf = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": f"{group_id}-{os.getpid()}-{int(time.time() * 1000)}",
            "auto.offset.reset": "latest",
            "socket.receive.buffer.bytes": 10 * 1024 * 1024,
            "fetch.message.max.bytes": 5 * 1024 * 1024,
            "max.partition.fetch.bytes": 5 * 1024 * 1024,
            "fetch.wait.max.ms": 5,
            "enable.auto.commit": False,
            "allow.auto.create.topics": True,
            "topic.metadata.refresh.interval.ms": 2000,
        }

        self._consumer = Consumer(conf)
        self._ensure_assigned()

    def _ensure_assigned(self) -> bool:
        if self._assigned:
            return True

        now = time.perf_counter()
        if (now - self._last_assign_attempt) < 1.0:
            return False
        self._last_assign_attempt = now

        try:
            md = self._consumer.list_topics(self.topic, timeout=1.5)
            topic_meta = md.topics.get(self.topic)
            if topic_meta is None or topic_meta.error is not None or not topic_meta.partitions:
                if not self._warned_no_topic:
                    logging.warning(
                        f"[Kafka] Tópico '{self.topic}' ainda não existe ou sem partições em "
                        f"'{self.bootstrap_servers}'. Aguardando produtores publicarem frames..."
                    )
                    self._warned_no_topic = True
                return False

            tps = []
            for pid in sorted(topic_meta.partitions.keys()):
                try:
                    low, high = self._consumer.get_watermark_offsets(
                        TopicPartition(self.topic, pid), timeout=1.5
                    )
                    # Start up to 32 messages before high watermark to immediately pick up
                    # the latest compacted frame per camera key without replaying old history
                    start_off = max(low, high - 32) if high > low else OFFSET_END
                except Exception:
                    start_off = OFFSET_END
                tps.append(TopicPartition(self.topic, pid, start_off))

            self._consumer.assign(tps)
            self._assigned = True
            logging.info(
                f"[Kafka] Conectado a '{self.bootstrap_servers}' | Tópico '{self.topic}' "
                f"({len(tps)} partição/ões) atribuído com sucesso."
            )
            return True
        except Exception as e:
            logging.warning(f"[Kafka] Falha ao conectar ao broker '{self.bootstrap_servers}': {e}")
            return False

    def _drain_and_decode(self) -> dict[str, np.ndarray]:
        if not self._ensure_assigned():
            time.sleep(0.02)
            return {}

        # Drain queued messages and keep only the newest JPEG payload per camera key
        msgs = self._consumer.consume(num_messages=64, timeout=0.005)
        if not msgs:
            return {}
        while True:
            more = self._consumer.consume(num_messages=64, timeout=0.0)
            if not more:
                break
            msgs.extend(more)

        latest_bytes_by_cam: dict[str, bytes] = {}
        for msg in msgs:
            if msg.error():
                continue
            val = msg.value()
            if val is None:
                continue
            key = msg.key()
            cam_id = (
                key.decode("utf-8", errors="replace")
                if key
                else (self.target_camera_id or "kafka_cam")
            )
            if self.target_camera_id and cam_id != self.target_camera_id:
                continue
            latest_bytes_by_cam[cam_id] = val

        if not latest_bytes_by_cam:
            return {}

        decoded: dict[str, np.ndarray] = {}
        for cam_id, img_bytes in latest_bytes_by_cam.items():
            frame = self._decode_frame(img_bytes)
            if frame is None:
                continue
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            decoded[cam_id] = frame

        return decoded

    def execute_batch(self) -> dict[str, np.ndarray]:
        """
        Drain and return the freshest decoded frame for every camera present in the Kafka batch.
        """
        batch: dict[str, np.ndarray] = {}
        while self._pending_frames:
            cid, frm = self._pending_frames.popleft()
            batch[cid] = frm
        batch.update(self._drain_and_decode())
        return batch

    def execute(self) -> Optional[Tuple[str, Any]]:
        if not self._pending_frames:
            decoded = self._drain_and_decode()
            for item in decoded.items():
                self._pending_frames.append(item)

        if not self._pending_frames:
            return None

        return self._pending_frames.popleft()

    @staticmethod
    def _decode_frame(img_bytes: bytes) -> Optional[np.ndarray]:
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.debug(f"Falha ao decodificar frame: {e}")
            return None

    def cleanup(self) -> None:
        if self._consumer:
            self._consumer.close()
