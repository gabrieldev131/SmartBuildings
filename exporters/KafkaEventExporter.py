# exporters/KafkaEventExporter.py
import json
import logging
from confluent_kafka import Producer
from exporters.ExporterRegistry import ExporterRegistry
from interfaces.IDataExporter import IDataExporter


@ExporterRegistry.register("kafka")
class KafkaEventExporter(IDataExporter):
    """Concrete `IDataExporter` publishing YOLO, DeepSort, Re-ID, and Threshold telemetry to Kafka topic 'Calma_Nathalia'."""

    def __init__(self, bootstrap_servers: str, topic: str = "Calma_Nathalia", stream_realtime: bool = True):
        self.topic = topic
        self.stream_realtime = stream_realtime
        self._producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "linger.ms": 5,
            "acks": 1,
        })
        logging.info(f"[KafkaEventExporter] Inicializado para tópico '{topic}' em '{bootstrap_servers}'.")

    @classmethod
    def from_config(cls, config) -> "KafkaEventExporter":
        servers = getattr(config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        topic = getattr(config, "KAFKA_EVENTS_TOPIC", "Calma_Nathalia")
        return cls(bootstrap_servers=servers, topic=topic)

    def _publish_dict(self, record: dict) -> None:
        try:
            payload_dict = {"kafka_topic": self.topic, **record}
            key = str(payload_dict.get("camera_id") or payload_dict.get("global_id") or "event").encode("utf-8")
            payload = json.dumps(payload_dict, ensure_ascii=False, default=str).encode("utf-8")
            self._producer.produce(self.topic, key=key, value=payload)
            self._producer.poll(0)
        except Exception as e:
            logging.debug(f"[KafkaEventExporter] Erro ao publicar evento no tópico '{self.topic}': {e}")

    def on_event(self, event: dict) -> None:
        if self.stream_realtime:
            self._publish_dict(event)

    def export(self, records: list[dict]) -> None:
        for rec in records:
            if not self.stream_realtime or rec.get("event_type") in ("CAMERA_PASSAGE", "AMBIENTE_SUMMARY"):
                self._publish_dict(rec)
        self._producer.flush(timeout=2.0)
        logging.info(f"[KafkaEventExporter] Dados sincronizados no tópico Kafka '{self.topic}'.")

    def cleanup(self) -> None:
        if self._producer is not None:
            self._producer.flush(timeout=2.0)
