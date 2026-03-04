# model/processing_worker.py
import cv2
import numpy as np
import logging

# Variaveis globais do processo worker. Carregadas uma vez por processo via initializer.
net = None
classes = []
output_layers = []

def initialize_worker(yolo_weights, yolo_cfg, yolo_names):
    """
    Inicializa o modelo YOLO dentro de cada processo do pool.
    Chamada automaticamente pelo multiprocessing.Pool via 'initializer'.
    CORRECAO: Usa logging em vez de print() para nao intercalar saidas
    de multiplos processos no console.
    """
    global net, classes, output_layers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [Worker] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.info("Carregando modelo YOLO no processo worker...")

    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

    # Opcional: descomenta para usar GPU via CUDA se disponivel
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    with open(yolo_names, "r") as f:
        classes = f.read().splitlines()

    output_layers = net.getUnconnectedOutLayersNames()
    logging.info("Modelo YOLO carregado com sucesso.")


def detect_people_in_frame(frame, confidence_threshold, nms_threshold):
    """
    Executa a inferencia YOLO em um frame e retorna apenas deteccoes de pessoas.

    Retorna: lista de tuplas (box, confidence) onde box = [x, y, w, h].

    NOTA DE PERFORMANCE: Esta funcao serializa o frame via pickle para enviar
    ao processo worker. Para resolucoes altas isso e caro. Mantenha
    PROCESSING_WIDTH/HEIGHT baixos no config (ex: 416x416 ou 640x480).
    """
    if net is None:
        logging.warning("Worker chamado antes de ser inicializado.")
        return []

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences = [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    if not boxes:
        return []

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indexes) == 0:
        return []

    return [(boxes[i], confidences[i]) for i in indexes.flatten()]