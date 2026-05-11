# vision/cameraWorker.py
import cv2
import time
import queue
import threading
from ultralytics import YOLO

from vision.featureExtractor import extract_color_histogram
from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
        
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea)

class CameraWorker(threading.Thread):
    def __init__(self, cam_id: str, input_queue: queue.Queue, config, global_manager, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"Worker-{cam_id}")
        self.cam_id = cam_id
        self.input_queue = input_queue
        self.config = config
        self.global_manager = global_manager
        self.stop_event = stop_event
        
        self.local_to_global_map = {}

    def run(self):
        model = YOLO(self.config.YOLO_MODEL_PATH, verbose=False)
        state_tracker = StoppedStateTracker(self.config)

        print(f"[{self.name}] Iniciado e a aguardar frames...")

        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            current_time = time.time()
            
            # MUDANÇA 1: Adicionado conf=0.40 e iou=0.50
            # Isso mata as "caixas tremidas" (fantasmas) que faziam o BoT-SORT perder e recriar IDs locais.
            results = model.track(
                frame, 
                classes=[0], 
                conf=0.40,      # Só confia em deteções fortes
                iou=0.50,       # Evita caixas duplas na mesma pessoa
                persist=True, 
                tracker="botsort.yaml", 
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                local_ids = results[0].boxes.id.cpu().numpy()
                
                is_occluded = [False] * len(boxes)
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        if compute_iou(boxes[i], boxes[j]) > 0.3:
                            is_occluded[i] = True
                            is_occluded[j] = True
                
                active_global_ids = set()
                for lid in local_ids:
                    if lid in self.local_to_global_map:
                        active_global_ids.add(self.local_to_global_map[lid])
                
                active_local_ids = set()
                
                for box, local_id, occluded in zip(boxes, local_ids, is_occluded):
                    active_local_ids.add(local_id)
                    feature_vector = None if occluded else extract_color_histogram(frame, box)
                    
                    if local_id not in self.local_to_global_map:
                        first_feature = feature_vector if feature_vector is not None else extract_color_histogram(frame, box)
                        
                        global_id = self.global_manager.get_or_create_global_id(
                            new_feature_vector=first_feature,
                            bbox=box,
                            cam_id=self.cam_id,
                            current_time=current_time,
                            active_global_ids=active_global_ids
                        )
                        self.local_to_global_map[local_id] = global_id
                        active_global_ids.add(global_id)
                    else:
                        global_id = self.local_to_global_map[local_id]
                        self.global_manager.update_existing_identity(
                            global_id=global_id,
                            new_feature_vector=feature_vector,
                            bbox=box,
                            cam_id=self.cam_id,
                            current_time=current_time
                        )
                    
                    is_stopped, elapsed = state_tracker.update_and_evaluate(global_id, box, current_time)
                    draw_person_annotation(frame, box, global_id, is_stopped, elapsed, self.config)

                lost_locals = [lid for lid in self.local_to_global_map if lid not in active_local_ids]
                for lid in lost_locals:
                    del self.local_to_global_map[lid]

            cv2.imshow(f"Camera {self.cam_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        print(f"[{self.name}] Encerrado.")