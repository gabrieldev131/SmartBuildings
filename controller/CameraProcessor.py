# controller/camera_processor.py
import cv2
import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from model.ProcessingWorker import detect_people_in_frame
# from model.Tracker import StatefulTracker <-- Removido, pois o DeepSORT + Dicionário farão esse papel

class CameraProcessor:
    def __init__(self, source_id, config):
        self.source_id = source_id
        self.config = config
        self.frame_counter = 0
        
        # Inicializando o DeepSORT
        self.tracker = DeepSort(
            max_age=30,
            n_init=1,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2
        )
        
        self.pending_detections = None
        
        # Dicionário para manter o histórico de movimento de cada ID
        # Formato: { track_id: {'last_center': (x, y), 'stopped_since': timestamp, 'is_stopped': False} }
        self.person_states = {}
        self.current_tracks = []
    def process_frame(self, frame, detection_pool):
        self.frame_counter += 1
        
        if self._is_detection_frame() and self.config.USE_MULTIPROCESSING and detection_pool:
            detection_pool.apply_async(
                detect_people_in_frame, 
                args=(frame, self.config.YOLO_CONFIDENCE_THRESHOLD, self.config.YOLO_NMS_THRESHOLD), 
                callback=self.update_trackers_from_detection
            )
        
        # --- A MÁGICA ACONTECE AQUI ---
        # Só atualizamos o DeepSORT se o callback do YOLO trouxe uma resposta nova.
        # (Seja ela vazia ou cheia). Se for None, apenas ignoramos e usamos a antiga.
        if self.pending_detections is not None:
            self.current_tracks = self.tracker.update_tracks(self.pending_detections, frame=frame)
            self.pending_detections = None # Reseta para esperar o próximo callback do YOLO
            
        boxes_and_states = []
        
        # Lemos sempre de self.current_tracks (que mantém as caixas na tela entre as detecções)
        for track in self.current_tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb() 
            
            x, y = int(ltrb[0]), int(ltrb[1])
            w, h = int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1])
            box = [x, y, w, h]
            
            is_stopped = self._check_if_stopped(track_id, box)
            boxes_and_states.append((box, is_stopped, track_id))

        processed_frame = self._draw_boxes(frame, boxes_and_states)
        
        self._cleanup_old_states([t.track_id for t in self.current_tracks if t.is_confirmed()])
        
        return processed_frame

    def _is_detection_frame(self):
        return self.frame_counter % self.config.DETECT_EVERY_N_FRAMES == 0

    def update_trackers_from_detection(self, yolo_results):
        """Prepara as detecções do YOLO para o DeepSORT."""
        formatted_detections = []
        # Assumindo que o yolo_results agora é uma lista de tuplas: (box, confidence)
        for box, conf in yolo_results:
            # DeepSORT exige: ([x, y, w, h], confidence, 'class_name')
            formatted_detections.append((box, conf, 'person'))
            
        self.pending_detections = formatted_detections

    def _check_if_stopped(self, track_id, box):
        """Verifica se um ID específico está parado com base no deslocamento do centro."""
        x, y, w, h = box
        current_center = (x + w/2, y + h/2)
        current_time = time.time()
        
        # Se for um ID novo, inicializa o estado
        if track_id not in self.person_states:
            self.person_states[track_id] = {
                'last_center': current_center,
                'stopped_since': current_time,
                'is_stopped': False
            }
            return False
            
        state = self.person_states[track_id]
        last_center = state['last_center']
        
        # Calcula a distância euclidiana que o centro da caixa se moveu
        distance = np.sqrt((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)
        
        if distance < self.config.STOPPED_PIXEL_THRESHOLD:
            # Pessoa não se moveu o suficiente, verifica o tempo
            time_stopped = current_time - state['stopped_since']
            if time_stopped >= self.config.STOPPED_SECONDS_THRESHOLD:
                state['is_stopped'] = True
        else:
            # Pessoa se moveu, reseta o cronômetro e a flag
            if distance > self.config.MOVEMENT_BREAKOUT_THRESHOLD:
                state['stopped_since'] = current_time
                state['is_stopped'] = False
                
        # Atualiza a última posição conhecida
        state['last_center'] = current_center
        
        return state['is_stopped']

    def _cleanup_old_states(self, active_track_ids):
        """Remove dicionários de pessoas que já saíram da tela para evitar vazamento de memória."""
        keys_to_delete = [tid for tid in self.person_states.keys() if tid not in active_track_ids]
        for tid in keys_to_delete:
            del self.person_states[tid]

    def _draw_boxes(self, frame, boxes_and_states):
        """Desenha caixas, ID e muda a cor se a pessoa estiver parada."""
        for box, is_stopped, track_id in boxes_and_states:
            color = self.config.PERSON_BOX_COLOR
            if is_stopped:
                color = self.config.STOPPED_BOX_COLOR
            
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Texto mostrando o ID e se está parado
            label = f"ID: {track_id}"
            if is_stopped:
                label += " - PARADO"
                
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return frame