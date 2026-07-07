# vision/cameraWorker.py
import cv2
import time
import queue
import threading
import numpy as np
from ultralytics import YOLO
import torch

# Importação do Deep SORT
from deep_sort_realtime.deepsort_tracker import DeepSort

from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation

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
        # YOLO apenas para deteção
        model = YOLO(self.config.YOLO_MODEL_PATH, task='detect', verbose=False)
        state_tracker = StoppedStateTracker(self.config)

        # CORREÇÃO 1: Ajuste rigoroso da supressão de não-máximos (NMS)
        tracker = DeepSort(
            max_age=300,
            n_init=6,
            nms_max_overlap=0.35,    # <-- MUDADO DE 1.0. Impede caixas sobrepostas na mesma pessoa
            max_cosine_distance=0.2,
            embedder="mobilenet",
            half=True,               
            bgr=True,
            embedder_gpu=True
        )

        print(f"[{self.name}] Iniciado com Deep SORT (Otimizado) a aguardar frames...")
        
        # Inicialização das variáveis para cálculo de FPS
        prev_time = time.time()
        fps = 0.0

        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=1.0)
                
            except queue.Empty:
                continue

            current_time = time.time()
            
            # Atualiza e calcula o FPS atual (com suavização para o número não piscar tanto)
            frame_time = current_time - prev_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
            fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps
            prev_time = current_time
            
            # CORREÇÃO 2: Adição do iou=0.45 no YOLO.
            # Isto impede que o YOLO envie um corpo e um rosto como duas pessoas diferentes.
            results = model.predict(
                frame, 
                classes=[0], 
                conf=0.50,       # Apenas deteções com mais de 50% de certeza
                iou=0.45,        # Corta duplicações nativas do YOLO
                verbose=False,
                device=0,      # <-- FORÇA O DISPOSITIVO (Impede verificações de hardware a cada frame)
                half=True,     # <-- FORÇA FP16 (Corta o tráfego no barramento PCIe pela metade)
                stream=True    # <-- REDUZ OVERHEAD DE MEMÓRIA (Gera um generator em vez de uma lista pesada)
            )
            
            bbs = []
            # Como stream=True retorna um generator, iteramos diretamente
            for result in results:
                if result.boxes is not None:
                    # Mover os tensores para a CPU em lote usando numpy() de forma vetorizada
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i]
                        w = x2 - x1
                        h = y2 - y1
                        bbs.append(([x1, y1, w, h], float(confs[i]), int(clss[i])))
            
            # O Deep SORT rastreia e extrai os vetores
            tracks = tracker.update_tracks(bbs, frame=frame)
            
            active_local_ids = set()
            active_global_ids = set()
            for tid in self.local_to_global_map:
                active_global_ids.add(self.local_to_global_map[tid])
            
            for track in tracks:
                
                # CORREÇÃO 3: Filtro anti "Caixas Fantasmas/Inchaço"
                # track.time_since_update > 0 significa que o YOLO não viu a pessoa neste frame exato.
                # Se isso acontecer, o Deep SORT está a usar o Filtro de Kalman para "adivinhar" onde ela está.
                # Nós ignoramos estas adivinhações para que a caixa não seja desenhada nem cresça do nada.
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
                is_stopped, elapsed = state_tracker.update_and_evaluate(global_id, box, current_time)
                draw_person_annotation(frame, box, global_id, is_stopped, elapsed, self.config)

            # Limpar lixo
            lost_locals = [lid for lid in self.local_to_global_map if lid not in active_local_ids]
            for lid in lost_locals:
                del self.local_to_global_map[lid]

            # --- RENDERIZAÇÃO DO HUD (Estatísticas na tela) ---
            q_size = self.input_queue.qsize()
            
            # Desenha um pequeno fundo semi-transparente para o texto ser legível em fundos brancos
            cv2.rectangle(frame, (5, 5), (160, 65), (0, 0, 0), -1)
            
            # Escreve o FPS e o Tamanho da Fila (amarelo)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Fila: {q_size}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            # -------------------------------------------------

            try:
                # Envia o frame final (com as caixas e IDs desenhados) para o router principal exibir
                self.display_queue.put_nowait((self.cam_id, frame))
            except queue.Full:
                pass
            #cv2.imshow(f"Camera {self.cam_id}", frame)
            """if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break"""

        print(f"[{self.name}] Encerrado.")