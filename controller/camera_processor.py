# controller/camera_processor.py
import cv2
from model.tracker import OpticalFlowTracker
from model.processing_worker import detect_people_in_frame
class CameraProcessor:
    def __init__(self, source_id, config):
        self.source_id = source_id
        self.config = config
        self.frame_counter = 0
        self.trackers = []
        self.last_gray_frame = None

    # controller/camera_processor.py
import cv2
from model.tracker import OpticalFlowTracker
from model.processing_worker import detect_people_in_frame # Importamos a função de detecção

class CameraProcessor:
    def __init__(self, source_id, config):
        self.source_id = source_id
        self.config = config
        self.frame_counter = 0
        self.trackers = []
        self.last_gray_frame = None

    # --- MÉTODO PÚBLICO PRINCIPAL (AGORA MAIS SIMPLES) ---
    def process_frame(self, frame, detection_pool):
        """
        Orquestra o processamento de um único frame.
        """
        self.frame_counter += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Dispara uma nova detecção em paralelo, se for a hora
        self._trigger_detection_if_needed(frame, detection_pool)

        # 2. Atualiza os rastreadores existentes e obtém suas posições
        tracked_boxes = self._update_trackers(gray_frame)
        
        # 3. Desenha as caixas no frame original
        processed_frame = self._draw_boxes(frame, tracked_boxes)

        # 4. Armazena o frame em escala de cinza para o próximo ciclo
        self.last_gray_frame = gray_frame
            
        return processed_frame

    # --- MÉTODOS PRIVADOS DE LÓGICA ---
    def _is_detection_frame(self):
        """Verifica se o frame atual é um frame de detecção."""
        return self.frame_counter % self.config.DETECT_EVERY_N_FRAMES == 0

    def _trigger_detection_if_needed(self, frame, detection_pool):
        """Envia o frame para o pool de detecção se for o momento certo."""
        if self._is_detection_frame() and self.config.USE_MULTIPROCESSING and detection_pool:
            detection_pool.apply_async(
                detect_people_in_frame, 
                args=(frame, self.config.YOLO_CONFIDENCE_THRESHOLD, self.config.YOLO_NMS_THRESHOLD), 
                # O callback usa uma lambda para passar o 'self' e o resultado
                callback=lambda result: self.update_trackers_from_detection(result)
            )

    def _update_trackers(self, gray_frame):
        """Atualiza todos os rastreadores ativos e retorna suas novas posições."""
        boxes_to_draw = []
        updated_trackers = []
        for tracker in self.trackers:
            (success, new_boxes) = tracker.update(gray_frame)
            if success and new_boxes:
                boxes_to_draw.extend(new_boxes)
                updated_trackers.append(tracker) # Mantém apenas os rastreadores bem-sucedidos
        self.trackers = updated_trackers
        return boxes_to_draw
    
    def _draw_boxes(self, frame, boxes):
        """Desenha uma lista de caixas em um frame."""
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.config.PERSON_BOX_COLOR, 2)
        return frame

    def update_trackers_from_detection(self, detected_boxes):
        """
        Callback: Chamado quando a detecção do YOLO termina.
        Cria novos rastreadores com base nas novas detecções.
        """
        new_trackers = []
        if self.last_gray_frame is not None:
            for box in detected_boxes:
                tracker = OpticalFlowTracker(self.last_gray_frame, box)
                new_trackers.append(tracker)
        
        # Substitui a lista de trackers antiga pela nova, recém-detectada
        self.trackers = new_trackers

    def update_trackers(self, detected_boxes):
        """
        Callback: Chamado quando a detecção do YOLO termina.
        Atualiza os rastreadores com as novas detecções.
        """
        updated_trackers = []
        if self.last_gray_frame is not None:
            for box in detected_boxes:
                # Cria um novo tracker para cada caixa detectada pelo YOLO
                tracker = OpticalFlowTracker(self.last_gray_frame, box)
                updated_trackers.append(tracker)
        
        # Substitui a lista de trackers antiga pela nova, recém-detectada
        self.trackers = updated_trackers