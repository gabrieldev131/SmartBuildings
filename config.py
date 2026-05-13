# config.py
class Config:
    # Modelos
    YOLO_MODEL_PATH = "models/yolo26n.pt"  # Atualizado caso esteja a usar v8
    
    # Limiares de Movimento / Paragem
    STOPPED_SECONDS_THRESHOLD = 3
    STOPPED_PIXEL_THRESHOLD = 15
    MOVEMENT_BREAKOUT_THRESHOLD = 30
    
    # === LIMIARES DE IDENTIDADE BIFURCADOS ===
    SIMILARITY_THRESHOLD = 0.20           # Rigoroso: Para rastreio dentro da MESMA câmara
    INTER_CAMERA_THRESHOLD = 0.35         # Permissivo: Para compensar mudanças de iluminação ao mudar de câmara
    
    MAX_SPATIAL_DISTANCE = 150    
    MAX_TIME_LOST = 10.0                 # Aumentado para 10s para permitir que a pessoa ande entre corredores sem câmara
    CAMERA_SWITCH_COOLDOWN = 2.0         # Evita ping-pong entre câmaras com sobreposição
    AUTO_CLUSTER_TIME_THRESHOLD = 3.0    # Tempo para considerar que duas câmaras filmam o mesmo ambiente
    
    # Interface
    COLOR_STOPPED = (0, 0, 255)  
    COLOR_MOVING = (0, 255, 0)   
    FONT_SCALE = 0.5

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    KAFKA_TOPIC = "meu-topico-de-video"
    KAFKA_GROUP_ID = "smart-builds-consumer"
    KAFKA_TARGET_CAMERA = None  
    KAFKA_EXPECTED_CAMERAS = 4

    # Capturas
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 480