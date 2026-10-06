# vision/cameraWorker.py
import cv2
import time
import queue
import threading
import numpy as np
from ultralytics import YOLO
import torch

# from deep_sort_realtime.deepsort_tracker import DeepSort

from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation

# 2. IMPORTAÇÃO DO EXTRATOR LEVE: Usaremos CPU e matemática simples para o Re-ID
from vision.featureExtractor import extract_color_histogram

class CameraWorker(threading.Thread):
    def __init__(self, cam_id: str, input_queue: queue.Queue, display_queue: queue.Queue, config, global_manager, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"Worker-{cam_id}")
        self.cam_id = cam_id
        self.input_queue = input_queue
        self.display_queue = display_queue
        self.config = config
        self.global_manager = global_manager
        self.stop_event = stop_event
        
        self.local_to_global_map = {}

    def run(self):
        torch.backends.cudnn.benchmark = True
        
        # O model permanece igual, mas na chamada (abaixo) usaremos .track() em vez de .predict()
        model = YOLO(self.config.YOLO_MODEL_PATH, task='detect', verbose=False)
        state_tracker = StoppedStateTracker(self.config)

        print(f"[{self.name}] Iniciado com ByteTrack Nativo a aguardar frames...")
        
        prev_time = time.time()
        fps = 0.0

        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            current_time = time.time()
            
            frame_time = current_time - prev_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
            fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps
            prev_time = current_time
            
            # 3. MUDANÇA PARA O RASTREADOR NATIVO:
            # Substituímos o model.predict por model.track e adicionamos persist e tracker
            results = model.track(
                frame, 
                classes=[0], 
                conf=0.50,       
                iou=0.45,        
                verbose=False,
                device=0,      
                half=True,     
                stream=True,
                persist=True,             # Mantém a memória dos IDs entre frames
                tracker="bytetrack.yaml"  # Aciona o ByteTrack nativo do YOLO
            )
            
            active_local_ids = set()
            active_global_ids = set()
            for tid in self.local_to_global_map:
                active_global_ids.add(self.local_to_global_map[tid])
            
            for result in results:
                # Se não houver detecções ou o rastreador ainda não atribuiu IDs, saltamos
                if result.boxes is None or result.boxes.id is None:
                    continue
                
                # Movemos tudo de forma vetorizada
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    local_id = track_ids[i]
                    box = [x1, y1, x2, y2]
                    
                    active_local_ids.add(local_id)
                    
                    # 4. EXTRAÇÃO LEVE: Substituímos o MobileNet pelo Histograma de Cores
                    feature_vector = extract_color_histogram(frame, box)
                    
                    # Registo de nova pessoa ou atualização
                    if local_id not in self.local_to_global_map:
                        if feature_vector is None:
                            continue 
                            
                        global_id = self.global_manager.get_or_create_global_id(
                            new_feature_vector=feature_vector,
                            bbox=box,
                            cam_id=self.cam_id,
                            current_time=current_time,
                            active_global_ids=active_global_ids
                        )
                        self.local_to_global_map[local_id] = global_id
                        active_global_ids.add(global_id)
                    else:
                        global_id = self.local_to_global_map[local_id]
                        if feature_vector is not None:
                            self.global_manager.update_existing_identity(
                                global_id=global_id,
                                new_feature_vector=feature_vector,
                                bbox=box,
                                cam_id=self.cam_id,
                                current_time=current_time
                            )
                    
                    # Interface Visual
                    is_stopped, elapsed = state_tracker.update_and_evaluate(global_id, box, current_time)
                    draw_person_annotation(frame, box, global_id, is_stopped, elapsed, self.config)

            # Limpar lixo
            lost_locals = [lid for lid in self.local_to_global_map if lid not in active_local_ids]
            for lid in lost_locals:
                del self.local_to_global_map[lid]

            # --- RENDERIZAÇÃO DO HUD ---
            q_size = self.input_queue.qsize()
            cv2.rectangle(frame, (5, 5), (160, 65), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Fila: {q_size}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            # ---------------------------

            try:
                self.display_queue.put_nowait((self.cam_id, frame))
            except queue.Full:
                pass

        print(f"[{self.name}] Encerrado.")