# extraction/frameReaderCommand/ReadKafkaCommand.py
from collections import deque
import logging
import os
import time
from typing import Any, Optional, Tuple
import cv2
import numpy as np
from confluent_kafka import Consumer, OFFSET_END, TopicPartition

from extraction.frameReaderCommand.IFrameCommand import IFrameCommand


class ReadKafkaCommand(IFrameCommand):
    """Concrete Command for consuming and decoding latest frames per camera key from Kafka (SRP & LSP)."""

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str, width: int, height: int, target_camera_id: Optional[str] = ""):
        self.bootstrap_servers, self.topic = bootstrap_servers, topic
        self.width, self.height = width, height
        self.target_camera_id = target_camera_id or ""
        self._pending_frames: deque[Tuple[str, np.ndarray]] = deque()
        self._assigned, self._warned_no_topic = False, False
        self._last_assign_attempt = 0.0
        self._consumer = Consumer(self._build_kafka_conf(bootstrap_servers, group_id))
        self._ensure_assigned()

    @staticmethod
    def _build_kafka_conf(bootstrap_servers: str, group_id: str) -> dict:
        return {
            "bootstrap.servers": bootstrap_servers,
            "group.id": f"{group_id}-{os.getpid()}-{int(time.time() * 1000)}",
            "auto.offset.reset": "latest", "socket.receive.buffer.bytes": 10 * 1024 * 1024,
            "fetch.message.max.bytes": 5 * 1024 * 1024, "max.partition.fetch.bytes": 5 * 1024 * 1024,
            "fetch.wait.max.ms": 5, "enable.auto.commit": False,
            "allow.auto.create.topics": True, "topic.metadata.refresh.interval.ms": 2000,
        }

    def _build_partition_assignments(self, partitions: dict) -> list[TopicPartition]:
        tps = []
        for pid in sorted(partitions.keys()):
            try:
                low, high = self._consumer.get_watermark_offsets(TopicPartition(self.topic, pid), timeout=1.5)
                start_off = max(low, high - 32) if high > low else OFFSET_END
            except Exception:
                start_off = OFFSET_END
            tps.append(TopicPartition(self.topic, pid, start_off))
        return tps

    def _ensure_assigned(self) -> bool:
        if self._assigned or (time.perf_counter() - self._last_assign_attempt) < 1.0:
            return self._assigned
        self._last_assign_attempt = time.perf_counter()
        try:
            meta = self._consumer.list_topics(self.topic, timeout=1.5).topics.get(self.topic)
            if meta is None or meta.error is not None or not meta.partitions:
                if not self._warned_no_topic:
                    logging.warning(f"[Kafka] Tópico '{self.topic}' indisponível em '{self.bootstrap_servers}'.")
                    self._warned_no_topic = True
                return False
            tps = self._build_partition_assignments(meta.partitions)
            self._consumer.assign(tps)
            self._assigned = True
            logging.info(f"[Kafka] Conectado a '{self.bootstrap_servers}' | Tópico '{self.topic}' ({len(tps)} partições).")
            return True
        except Exception as e:
            logging.warning(f"[Kafka] Falha ao conectar a '{self.bootstrap_servers}': {e}")
            return False

    def _consume_latest_bytes_by_cam(self) -> dict[str, bytes]:
        msgs = self._consumer.consume(num_messages=64, timeout=0.005)
        while msgs:
            more = self._consumer.consume(num_messages=64, timeout=0.0)
            if not more:
                break
            msgs.extend(more)
        latest: dict[str, bytes] = {}
        for m in msgs or []:
            if m.error() or m.value() is None:
                continue
            cam_id = m.key().decode("utf-8", errors="replace") if m.key() else (self.target_camera_id or "kafka_cam")
            if not self.target_camera_id or cam_id == self.target_camera_id:
                latest[cam_id] = m.value()
        return latest

    def _drain_and_decode(self) -> dict[str, np.ndarray]:
        if not self._ensure_assigned():
            time.sleep(0.02)
            return {}
        decoded: dict[str, np.ndarray] = {}
        for cam_id, img_bytes in self._consume_latest_bytes_by_cam().items():
            frame = self._decode_frame(img_bytes)
            if frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                decoded[cam_id] = frame
        return decoded

    def execute_batch(self) -> dict[str, np.ndarray]:
        batch: dict[str, np.ndarray] = {}
        while self._pending_frames:
            cid, frm = self._pending_frames.popleft()
            batch[cid] = frm
        batch.update(self._drain_and_decode())
        return batch

    def execute(self) -> Optional[Tuple[str, Any]]:
        if not self._pending_frames:
            self._pending_frames.extend(self._drain_and_decode().items())
        return self._pending_frames.popleft() if self._pending_frames else None

    @staticmethod
    def _decode_frame(img_bytes: bytes) -> Optional[np.ndarray]:
        try:
            return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None

    def cleanup(self) -> None:
        if self._consumer:
            self._consumer.close()
