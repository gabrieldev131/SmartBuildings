# vision/cameraWorker.py
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch

# Importação do Deep SORT
from deep_sort_realtime.deepsort_tracker import DeepSort

from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation

class CameraWorker:
    """
    Worker síncrono responsável por executar inferência (YOLO) e rastreio (DeepSORT)
    numa única thread de forma sequencial.
    """
    def __init__(self, cam_id: str, config, global_manager):
        self.cam_id = cam_id
        self.config = config
        self.global_manager = global_manager
        
        self.local_to_global_map = {}
        
        # Otimizações globais do PyTorch
        torch.backends.cudnn.benchmark = True
        
        # YOLO apenas para deteção instanciado uma única vez
        self.model = YOLO(self.config.YOLO_MODEL_PATH, task='detect', verbose=False)
        self.state_tracker = StoppedStateTracker(self.config)

        # Configuração do Deep SORT (Otimizado com as suas configurações)
        self.tracker = DeepSort(
            max_age=300,
            n_init=6,
            nms_max_overlap=0.35,
            max_cosine_distance=0.2,
            embedder="mobilenet",
            half=True,               
            bgr=True,
            embedder_gpu=True
        )

        # Variáveis de cálculo de FPS transferidas para o escopo da classe
        self.prev_time = time.time()
        self.fps = 0.0

        print(f"[Worker-{self.cam_id}] Inicializado de forma síncrona com Deep SORT (Otimizado).")

    def process_frame(self, frame):
        """
        Recebe um frame bruto, aplica as deteções, desenha as anotações
        e retorna o frame processado para exibição.
        """
        current_time = time.time()
        
        # Atualiza e calcula o FPS atual (com suavização para o número não piscar tanto)
        frame_time = current_time - self.prev_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps
        self.prev_time = current_time
        
        # Inferência do YOLO
        results = self.model.predict(
            frame, 
            classes=[0], 
            conf=0.50,       
            iou=0.45,        
            verbose=False,
            device=0,      
            half=True,     
            stream=True    
        )
        
        bbs = []
        for result in results:
            # Verificação de segurança adicional para evitar processar tensores vazios
            if result.boxes is not None and len(result.boxes) > 0:
                
                # 1. Extração para CPU e conversão para NumPy em lote
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()
                
                # 2. Vetorização Matemática (NumPy processa tudo de uma vez a nível de C)
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                w = boxes[:, 2] - x1
                h = boxes[:, 3] - y1
                
                # 3. Zip com Generator: Monta as tuplas drasticamente mais rápido
                bbs.extend(
                    ([float(bx), float(by), float(bw), float(bh)], float(c), int(cls_id))
                    for bx, by, bw, bh, c, cls_id in zip(x1, y1, w, h, confs, clss)
                )
        
        # O Deep SORT rastreia e extrai os vetores
        tracks = self.tracker.update_tracks(bbs, frame=frame)
        
        active_local_ids = set()
        active_global_ids = set()
        for tid in self.local_to_global_map:
            active_global_ids.add(self.local_to_global_map[tid])
        
        for track in tracks:
            # Filtro anti "Caixas Fantasmas/Inchaço"
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
                
            local_id = track.track_id
            active_local_ids.add(local_id)
            
            ltrb = track.to_ltrb()
            box = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
            
            feature_vector = None
            if track.features and len(track.features) > 0:
                raw_feature = np.array(track.features[-1])
                feature_vector = raw_feature / np.linalg.norm(raw_feature)
            
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
            is_stopped, elapsed = self.state_tracker.update_and_evaluate(global_id, box, current_time)
            draw_person_annotation(frame, box, global_id, is_stopped, elapsed, self.config)

        # Limpar lixo
        lost_locals = [lid for lid in self.local_to_global_map if lid not in active_local_ids]
        for lid in lost_locals:
            del self.local_to_global_map[lid]

        # --- RENDERIZAÇÃO DO HUD (Estatísticas na tela) ---
        # A altura do retângulo preto foi reduzida já que a fila não é mais exibida
        cv2.rectangle(frame, (5, 5), (160, 40), (0, 0, 0), -1)
        
        # Escreve o FPS (amarelo)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        # -------------------------------------------------

        return frame

    def cleanup(self):
        """Método opcional para limpar recursos se necessário durante o shutdown."""
        print(f"[Worker-{self.cam_id}] Recursos libertados.")