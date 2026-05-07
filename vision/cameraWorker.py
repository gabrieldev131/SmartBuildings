# vision/camera_worker.py
import cv2
import time
import queue
import threading
from ultralytics import YOLO

from vision.featureExtractor import extract_color_histogram
from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation

class CameraWorker(threading.Thread):
    """
    Thread dedicada ao processamento de uma única câmara.
    Consome frames de uma fila injetada pelo AppController.
    """
    def __init__(self, cam_id: str, input_queue: queue.Queue, config, global_manager, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"Worker-{cam_id}")
        self.cam_id = cam_id
        self.input_queue = input_queue
        self.config = config
        self.global_manager = global_manager
        self.stop_event = stop_event

    def run(self):
        # Cada câmara tem o seu próprio rastreador YOLO para manter o estado interno
        model = YOLO(self.config.YOLO_MODEL_PATH, verbose=False)
        state_tracker = StoppedStateTracker(self.config)

        print(f"[{self.name}] Iniciado e aguardando frames...")

        while not self.stop_event.is_set():
            try:
                # Aguarda até 1 segundo por um novo frame na fila
                frame = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            current_time = time.time()
            
            # Tracking nativo (ByteTrack recomendado para manter IDs em oclusões)
            results = model.track(
                frame, 
                classes=[0], 
                persist=True, 
                tracker="bytetrack.yaml", 
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                local_ids = results[0].boxes.id.cpu().numpy()
                
                for box, local_id in zip(boxes, local_ids):
                    # 1. Extração
                    feature_vector = extract_color_histogram(frame, box)
                    
                    # 2. Re-Identificação Global (Manager)
                    global_id = self.global_manager.get_or_create_global_id(
                        new_feature_vector=feature_vector,
                        bbox=box,
                        cam_id=self.cam_id,
                        current_time=current_time
                    )
                    
                    # 3. Estado de Paragem
                    is_stopped, elapsed = state_tracker.update_and_evaluate(global_id, box, current_time)

                    # 4. Desenho (UI)
                    draw_person_annotation(frame, box, global_id, is_stopped, elapsed, self.config)

            # Exibição
            cv2.imshow(f"Camera {self.cam_id}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        print(f"[{self.name}] Encerrado.")