# config.py

# --- Configurações de Rede ---
NETWORK_BASE = "10.145.80"
IP_RANGE_TO_SCAN = range(1, 255) 
RTSP_PORT = 554

# --- Credenciais das Câmeras ---
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "Aluno@00" 

# --- Configurações de Multiprocessos ---
# Habilita o uso de múltiplos processos para a detecção.
USE_MULTIPROCESSING = True
# Número de processos trabalhadores. None usa o número de núcleos da CPU.
NUM_WORKER_PROCESSES = 1
# Roda a detecção pesada a cada N frames. Nos outros, usa o rastreamento.
DETECT_EVERY_N_FRAMES = 15

# Tipo de rastreador a ser usado pelo OpenCV. CSRT é preciso, KCF é mais rápido.
TRACKER_TYPE = "CSRT" # Opções: "CSRT", "KCF", "MOSSE"

# --- Configurações de Captura ---
CAPTURE_INTERVAL_SECONDS = 60 # 1 minuto
SAVE_DIRECTORY = "capturas_mvc"
STREAM_TYPE = "102" # 101 para Main Stream, 102 para Sub Stream
BUFFER_SIZE = 10
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 480

# --- Configurações do Modelo de Detecção YOLOv3 ---
YOLO_WEIGHTS = "dnn_model/yolov3.weights"
YOLO_CFG = "dnn_model/yolov3.cfg"
YOLO_NAMES = "dnn_model/coco.names"
YOLO_CONFIDENCE_THRESHOLD = 0.5 # Limite de confiança para detecção
YOLO_NMS_THRESHOLD = 0.4      # Limite para a Supressão Não Máxima (NMS)

# --- Configurações do Modelo de Detecção de Pessoas (DNN) ---
DNN_MODEL_PROTO = "dnn_model/MobileNetSSD_deploy.prototxt"
DNN_MODEL_WEIGHTS = "dnn_model/MobileNetSSD_deploy.caffemodel"
DNN_CONFIDENCE_THRESHOLD = 0.5 # Limite de confiança para detecção (0.5 = 50%)

# Classes que o modelo pode detectar. A classe 'person' é a de interesse.
# O modelo MobileNet SSD é treinado no dataset COCO.
DNN_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "day", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

# A ID da classe 'person' no array acima (índice 15)
PERSON_CLASS_ID = 15 

# Cor do contorno do quadrado (BGR - Azul, Verde, Vermelho)
PERSON_BOX_COLOR = (0, 255, 0) # Verde