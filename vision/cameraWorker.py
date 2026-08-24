# vision/cameraWorker.py
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Importação do Deep SORT
from deep_sort_realtime.deepsort_tracker import DeepSort

from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation
from vision.featureExtractor import extract_color_histogram


class CameraWorker:
    """
    Worker síncrono responsável por executar inferência (YOLO) e rastreio (DeepSORT)
    numa única thread de forma sequencial.

    OTIMIZAÇÃO PRINCIPAL: o DeepSORT é instanciado com embedder=None. Em vez de deixar
    o DeepSORT rodar a sua própria rede neural (MobileNet) para gerar o embedding de
    aparência de cada pessoa a cada frame, nós fornecemos o NOSSO histograma de cor
    (extract_color_histogram) como embedding via `embeds=`. O mesmo vetor é guardado
    na track via `others=` e recuperado depois com `track.get_det_supplementary()`.

    Resultado: uma única extração de feature por pessoa por frame (em vez de duas:
    a mobilenet interna do DeepSORT + o histograma que já era calculado para o
    Global Re-ID). Isso remove por completo um forward pass de rede neural por pessoa
    por frame, que era o maior responsável pelo aquecimento sustentado da CPU/GPU.
    """
    def __init__(self, cam_id: str, config, global_manager):
        self.cam_id = cam_id
        self.config = config
        self.global_manager = global_manager

        self.local_to_global_map = {}

        # YOLO apenas para deteção instanciado uma única vez
        self.model = YOLO(self.config.YOLO_MODEL_PATH, task='detect', verbose=False)
        self.state_tracker = StoppedStateTracker(self.config)

        # embedder=None: eliminamos o custo da mobilenet interna do DeepSORT.
        # O custo de aparência do algoritmo passa a usar o nosso próprio histograma,
        # fornecido a cada chamada de update_tracks via "embeds".
        self.tracker = DeepSort(
            max_age=300,
            n_init=6,
            nms_max_overlap=0.35,
            max_cosine_distance=0.2,
            embedder=None,
        )

        # Controle de throttle: só reenviamos o vetor de aparência (EMA) para o
        # GlobalIdentityManager a cada N frames. Posição/tempo continuam sendo
        # atualizados TODO frame (isso é barato); o que é caro é a extração/blend
        # do histograma, e essa não precisa acontecer 30x/s por pessoa.
        self.feature_update_interval = getattr(config, 'FEATURE_UPDATE_INTERVAL', 3)
        self.frame_counter = 0

        # Variáveis de cálculo de FPS
        self.prev_time = time.time()
        self.fps = 0.0

        print(f"[Worker-{self.cam_id}] Inicializado (embedder próprio, sem MobileNet redundante).")

    def process_frame(self, frame):
        """
        Recebe um frame bruto, aplica as deteções, desenha as anotações
        e retorna o frame processado para exibição.
        """
        current_time = time.time()
        self.frame_counter += 1

        # Atualiza e calcula o FPS atual (com suavização)
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
            stream=True
        )

        bbs = []
        det_features = []  # um histograma por deteção, na mesma ordem de bbs

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()

                for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
                    w = x2 - x1
                    h = y2 - y1
                    bbs.append(([float(x1), float(y1), float(w), float(h)], float(c), int(cls_id)))

                    # Única extração de aparência do frame para esta deteção.
                    raw_feature = extract_color_histogram(frame, [x1, y1, x2, y2])
                    norm = np.linalg.norm(raw_feature)
                    feature_vector = raw_feature / norm if norm > 0 else raw_feature
                    det_features.append(feature_vector)

        # Passamos nosso histograma como "embeds" (custo de aparência do DeepSORT)
        # e também como "others" (para recuperar depois, sem recalcular nada).
        # Não passamos "frame=" pois não há embedder interno para rodar sobre ele.
        tracks = self.tracker.update_tracks(bbs, embeds=det_features, others=det_features)

        active_local_ids = set()
        active_global_ids = set(self.local_to_global_map.values())

        do_feature_update = (self.frame_counter % self.feature_update_interval == 0)

        for track in tracks:
            # Filtro anti "Caixas Fantasmas/Inchaço"
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            local_id = track.track_id
            active_local_ids.add(local_id)

            ltrb = track.to_ltrb()
            box = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]

            # Reaproveita a feature já calculada nesta rodada (zero recomputo).
            feature_vector = track.get_det_supplementary()

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

                # Throttle: o vetor de aparência (EMA) só é reenviado a cada N frames.
                # Posição/câmara/tempo são sempre atualizados (passando feature=None),
                # o que já é suportado nativamente por GlobalIdentity.update().
                sent_feature = feature_vector if do_feature_update else None
                self.global_manager.update_existing_identity(
                    global_id=global_id,
                    new_feature_vector=sent_feature,
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
        cv2.rectangle(frame, (5, 5), (160, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        # -------------------------------------------------

        return frame

    def cleanup(self):
        """Método opcional para limpar recursos se necessário durante o shutdown."""
        print(f"[Worker-{self.cam_id}] Recursos libertados.")