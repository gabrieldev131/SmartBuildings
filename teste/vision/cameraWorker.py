# vision/camera_worker.py
import cv2
import time
from ultralytics import YOLO

from vision.featureExtractor import extract_color_histogram
from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation

def process_camera_stream(cam_id, source, config, global_manager):
    model = YOLO(config.YOLO_MODEL_PATH, verbose=False)
    cap = cv2.VideoCapture(source)
    state_tracker = StoppedStateTracker(config)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        resized_frame = cv2.resize(frame, (640, 480))

        # USAMOS O BYTETRACK: Ele é muito mais robusto para manter IDs locais 
        # mesmo quando a pessoa está parcialmente escondida (low confidence).
        results = model.track(
            resized_frame, 
            classes=[0], 
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            local_ids = results[0].boxes.id.cpu().numpy() # YOLO ID
            
            for box, local_id in zip(boxes, local_ids):
                # 1. Extração de Identidade
                feature_vector = extract_color_histogram(resized_frame, box)
                
                # 2. Identidade Global (Agora passa a Bounding Box)
                global_id = global_manager.get_or_create_global_id(
                    new_feature_vector=feature_vector,
                    bbox=box, # <--- ENVIAMOS O RETÂNGULO
                    cam_id=cam_id,
                    current_time=current_time
                )
                
                # 3. Estado de Paragem
                is_stopped, elapsed = state_tracker.update_and_evaluate(global_id, box, current_time)

                # 4. Desenho
                draw_person_annotation(resized_frame, box, global_id, is_stopped, elapsed, config)

        cv2.imshow(f"Camera {cam_id}", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()