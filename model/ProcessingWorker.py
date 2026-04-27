# model/processing_worker.py
import logging
from ultralytics import YOLO

# Variavel global do processo worker. Carregada uma vez por processo via initializer.
model = None

def initialize_worker(model_path="yolov8s.pt"):
    """
    Inicializa o modelo YOLO (Ultralytics) dentro de cada processo do pool.
    Chamada automaticamente pelo multiprocessing.Pool via 'initializer'.
    """
    global model
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [Worker] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.info(f"Carregando modelo YOLO (Ultralytics) de {model_path} no processo worker...")

    # Instancia o modelo da Ultralytics. Ele fará o download automático 
    # se o arquivo .pt não existir localmente.
    model = YOLO(model_path)
    logging.info("Modelo YOLO carregado com sucesso.")


def detect_people_in_frame(frame, confidence_threshold, nms_threshold):
    """
    Executa a inferencia YOLO usando a biblioteca ultralytics e retorna apenas 
    deteccoes de pessoas.

    O AppController/CameraProcessor original espera o formato:
    Retorna: lista de tuplas (box, confidence) onde box = [x, y, w, h].
    """
    if model is None:
        logging.warning("Worker chamado antes de ser inicializado.")
        return []

    # O parametro classes=[0] diz à ultralytics para filtrar SOMENTE pessoas.
    # O parametro iou=nms_threshold ajusta o Non-Maximum Suppression interno.
    results = model(
        frame, 
        stream=True, 
        classes=[0], 
        conf=confidence_threshold, 
        iou=nms_threshold, 
        verbose=False
    )

    formatted_detections = []

    for r in results:
        for box in r.boxes:
            # Pega as coordenadas e confiança da Ultralytics
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()

            # Converte para o formato [x_canto_superior_esquerdo, y, largura, altura]
            # que é o formato exigido pelo pipeline do DeepSORT legado no CameraProcessor
            w = int(x2 - x1)
            h = int(y2 - y1)
            x = int(x1)
            y = int(y1)

            formatted_detections.append(([x, y, w, h], float(conf)))

    return formatted_detections