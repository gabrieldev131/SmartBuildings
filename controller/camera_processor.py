# controller/camera_processor.py
import cv2
from model.tracker import StatefulTracker # <-- MUDANÇA AQUI
from model.processing_worker import detect_people_in_frame

class CameraProcessor:
    def __init__(self, source_id, config):
        self.source_id = source_id
        self.config = config
        self.frame_counter = 0
        self.trackers = [] # Agora será uma lista de StatefulTrackers
        self.last_gray_frame = None

    def process_frame(self, frame, detection_pool):
        self.frame_counter += 1
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        
        if self._is_detection_frame() and self.config.USE_MULTIPROCESSING and detection_pool:
            detection_pool.apply_async(
                detect_people_in_frame, 
                args=(frame, self.config.YOLO_CONFIDENCE_THRESHOLD, self.config.YOLO_NMS_THRESHOLD), 
                callback=self.update_trackers_from_detection
            )
        
        # O método _update_trackers foi integrado aqui para simplicidade
        boxes_and_states = []
        updated_trackers = []
        for tracker in self.trackers:
            (success, box, is_stopped) = tracker.update(gray_frame, 
                                                        self.config.STOPPED_PIXEL_THRESHOLD, 
                                                        self.config.MOVEMENT_BREAKOUT_THRESHOLD , 
                                                        self.config.STOPPED_SECONDS_THRESHOLD, 
                                                        self.config.BOX_SMOOTHING_FACTOR)
            if success:
                boxes_and_states.append((box, is_stopped))
                updated_trackers.append(tracker)
        self.trackers = updated_trackers
        
        processed_frame = self._draw_boxes(frame, boxes_and_states)
        self.last_gray_frame = gray_frame
        return processed_frame

    def _is_detection_frame(self):
        return self.frame_counter % self.config.DETECT_EVERY_N_FRAMES == 0

    def _draw_boxes(self, frame, boxes_and_states):
        """Desenha caixas, mudando a cor se a pessoa estiver parada."""
        for box, is_stopped in boxes_and_states:
            color = self.config.PERSON_BOX_COLOR
            if is_stopped:
                color = self.config.STOPPED_BOX_COLOR
            
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if is_stopped:
                cv2.putText(frame, "PARADO", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def update_trackers_from_detection(self, detected_boxes):
        """Cria novos StatefulTrackers com base nas detecções do YOLO."""
        # NOTA: Esta é uma implementação simples. Uma versão mais avançada tentaria
        # corresponder as novas detecções com os trackers existentes (data association).
        new_trackers = []
        if self.last_gray_frame is not None:
            for box in detected_boxes:
                # Usamos a nova classe StatefulTracker
                tracker = StatefulTracker(self.last_gray_frame, box)
                new_trackers.append(tracker)
        self.trackers = new_trackers