import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 1. Inicializa o detector YOLOv8 (baixa o modelo 'nano' automaticamente)
model = YOLO("yolov8n.pt") 

# 2. Inicializa o rastreador DeepSORT
# max_age: quantos frames o tracker espera antes de "esquecer" um ID perdido
tracker = DeepSort(max_age=60, n_init=3)

# 3. Carrega o vídeo de teste (pode ser substituído por um link RTSP de câmera)
cap = cv2.VideoCapture("rtsp://admin:Aluno@00@10.145.80.51:554")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Roda a detecção focada apenas na classe 0 (pessoas)
    results = model(frame, stream=True, classes=[0], verbose=False)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extrai as coordenadas e a confiança da detecção
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            
            # O deep-sort-realtime exige o formato: ([x, y, largura, altura], confiança, classe)
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, 'person'))

    # 4. Atualiza o DeepSORT com as detecções do frame atual
    tracks = tracker.update_tracks(detections, frame=frame)

    # 5. Desenha os IDs e as caixas na imagem
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb() # Left, top, right, bottom
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Desenha a caixa delimitadora e o texto do ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Exibe o frame processado na tela
    cv2.imshow("Teste DeepSORT", frame)

    # Nota: Se em vez de exibir, você for processar e enviar as imagens adiante:
    # ret, buffer = cv2.imencode('.jpg', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()