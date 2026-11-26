import json
from confluent_kafka import Consumer
from Config import Config

consumer = Consumer({
    "bootstrap.servers": Config.KAFKA_BOOTSTRAP_SERVERS,  # e.g., "10.145.80.61:9092"
    "group.id": "calma-nathalia-reader",
    "auto.offset.reset": "latest",  # Use "earliest" to read all historical messages from the beginning
})

consumer.subscribe([Config.KAFKA_EVENTS_TOPIC])  # "Calma_Nathalia"
print(f"A aguardar mensagens do tópico '{Config.KAFKA_EVENTS_TOPIC}'...")

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            print("Sem mensagens recebidas.")
            continue
        if msg.error():
            print(f"Erro ao consumir mensagem: {msg.error()}")
            continue

        key = msg.key().decode("utf-8") if msg.key() else ""
        data = json.loads(msg.value().decode("utf-8"))

        # Access any top-level or nested YOLO / DeepSort / Threshold fields:
        cam_id = data.get("camera_id")
        people_count = data.get("people_in_camera")
        obj_class = data.get("object_class")
        limiares = data.get("limiares_percentuais", {})
        yolo_info = data.get("yolo", {})
        deepsort_info = data.get("deepsort", {})

        if cam_id == "cam2":
            print(
                data.get("deepsort", {}).get("age")
            )
except KeyboardInterrupt:
    pass
finally:
    consumer.close()
