# model/processing_worker.py
import cv2
import numpy as np

# Variáveis globais para o processo trabalhador. O modelo é carregado uma vez por processo.
net = None
classes = []
output_layers = []

def initialize_worker(yolo_weights, yolo_cfg, yolo_names):
    """Função para inicializar o modelo em cada processo do pool."""
    global net, classes, output_layers
    print("Inicializando um processo trabalhador e carregando o modelo YOLO...")
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    with open(yolo_names, "r") as f:
        classes = f.read().splitlines()
    output_layers_names = net.getUnconnectedOutLayersNames()
    output_layers = output_layers_names

def detect_people_in_frame(frame, confidence_threshold, nms_threshold):
    """
    Executa a detecção YOLO em um único frame. Esta função será chamada pelos processos.
    Retorna uma lista de caixas delimitadoras (x, y, w, h).
    """
    if net is None:
        return []

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
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
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    final_boxes = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            final_boxes.append(boxes[i])
            
    return final_boxes