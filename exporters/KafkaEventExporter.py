# exporters/KafkaEventExporter.py
import json
import logging
from confluent_kafka import Producer
from exporters.ExporterRegistry import ExporterRegistry
from interfaces.IDataExporter import IDataExporter


@ExporterRegistry.register("kafka")
class KafkaEventExporter(IDataExporter):
    """Concrete `IDataExporter` that publishes tracking events and summaries as JSON to a Kafka topic."""

    def __init__(self, bootstrap_servers: str, topic: str, stream_realtime: bool = True):
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
        topic = getattr(config, "KAFKA_EVENTS_TOPIC", "smartbuildings-eventos")
        return cls(bootstrap_servers=servers, topic=topic)

    def _publish_dict(self, record: dict) -> None:
        try:
            key = str(record.get("camera_id") or record.get("global_id") or "event").encode("utf-8")
            payload = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
            self._producer.produce(self.topic, key=key, value=payload)
            self._producer.poll(0)
        except Exception as e:
            logging.debug(f"[KafkaEventExporter] Erro ao publicar evento: {e}")

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
