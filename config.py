# config.py
class Config:
    # Modelos
    YOLO_MODEL_PATH = "models/yolo26n.pt"
    
    # Limiares de Movimento / Paragem
    STOPPED_SECONDS_THRESHOLD = 3
    STOPPED_PIXEL_THRESHOLD = 15
    MOVEMENT_BREAKOUT_THRESHOLD = 30
    
    # Limiares de Identidade
    SIMILARITY_THRESHOLD = 0.25      
    MAX_SPATIAL_DISTANCE = 150    
    MAX_TIME_LOST = 5.0
    
    # Interface
    COLOR_STOPPED = (0, 0, 255)  # Vermelho
    COLOR_MOVING = (0, 255, 0)   # Verde
    FONT_SCALE = 0.5

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    # Topico onde o produtor publica os frames JPEG das cameras.
    KAFKA_TOPIC = "meu-topico-de-video"
    # Consumer group. Mude se quiser duas instancias independentes lendo o mesmo topico.
    KAFKA_GROUP_ID = "smart-builds-consumer"
    # (Opcional) Se definido, o KafkaFrameReader ignora todas as cameras exceto esta.
    # None = consome todas as cameras presentes no topico.
    KAFKA_TARGET_CAMERA = None  # Ex: "cam3"
    # Estimativa de cameras no topico. Usado para dimensionar a fila interna.
    # Nao precisa ser exato: errar para cima e seguro.
    KAFKA_EXPECTED_CAMERAS = 4

    # Capturas
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 480

    CAMERA_SWITCH_COOLDOWN = 2.0  # Tempo mínimo (em segundos) entre mudanças de câmera para o mesmo ID global.