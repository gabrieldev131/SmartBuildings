# vision/cameraWorker.py
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Importação do Deep SORT
from deep_sort_realtime.deepsort_tracker import DeepSort

from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation
from vision.featureExtractor import extract_color_histogram_fast, extract_color_histogram_rich


class CameraWorker:
    """
    Worker responsável por executar inferência (YOLO) e rastreio (DeepSORT).

    ESTRATÉGIA DE FEATURES EM DUAS CAMADAS (importante para o Re-ID entre
    câmaras funcionar corretamente):

    1) BARATA (extract_color_histogram_fast) -- calculada para TODA deteção,
       TODO frame. Alimenta só o custo de aparência interno do DeepSORT
       (embeds/others), usado para associar caixas dentro da MESMA câmara
       quadro-a-quadro. Não precisa ser muito discriminativa: o Kalman
       filter + IOU do próprio DeepSORT já ajudam bastante aqui.

    2) RICA (extract_color_histogram_rich) -- calculada só nos dois
       momentos que REALMENTE decidem a identidade global de uma pessoa:
         a) quando um local_id novo aparece (é aí que perguntamos ao
            GlobalIdentityManager "quem é essa pessoa?" -- decisão
            cross-camera);
         b) na atualização periódica (throttled, a cada N frames) do vetor
            EMA guardado na identidade global.
       Como isso acontece raramente por pessoa (não a cada frame), pagar o
       custo de mais bins/mais resolução aqui não pesa na CPU, mas devolve
       a precisão de reconhecimento que se perde usando só a versão barata.

    Usar a versão barata nesses dois pontos foi o que reintroduziu o bug de
    "mesma pessoa, ID diferente em cada câmara" -- a versão rica é o que
    resolve isso mantendo o ganho de performance.
    """
    def __init__(self, cam_id: str, config, global_manager):
        self.cam_id = cam_id
        self.config = config
        self.global_manager = global_manager

        self.local_to_global_map = {}

        self.model = YOLO(self.config.YOLO_MODEL_PATH, task='detect', verbose=False)
        self.state_tracker = StoppedStateTracker(self.config)

        self.tracker = DeepSort(
            max_age=60,
            n_init=3,
            nms_max_overlap=0.35,
            max_cosine_distance=0.45,
            embedder=None,
        )

        self.feature_update_interval = getattr(config, 'FEATURE_UPDATE_INTERVAL', 3)
        self.frame_counter = 0

        self.prev_time = time.time()
        self.fps = 0.0

        # Prova de que todas as câmaras compartilham a MESMA instância do
        # GlobalIdentityManager (memória compartilhada entre threads/câmaras).
        # Se rodar com várias câmaras, compare esse id() nos logs -- deve
        # ser IDÊNTICO em todas elas.
        print(f"[Worker-{self.cam_id}] Inicializado. GlobalIdentityManager compartilhado id={id(self.global_manager)}")

    def process_frame(self, frame):
        current_time = time.time()
        self.frame_counter += 1

        frame_time = current_time - self.prev_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps
        self.prev_time = current_time

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
        boxes_xyxy = []      # guardamos as caixas originais para recalcular a feature rica depois
        det_features_fast = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()

                for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
                    w = x2 - x1
                    h = y2 - y1
                    box_xyxy = [float(x1), float(y1), float(x2), float(y2)]
                    bbs.append(([float(x1), float(y1), float(w), float(h)], float(c), int(cls_id)))
                    boxes_xyxy.append(box_xyxy)

                    raw_feature = extract_color_histogram_fast(frame, box_xyxy)
                    norm = np.linalg.norm(raw_feature)
                    det_features_fast.append(raw_feature / norm if norm > 0 else raw_feature)

        # embeds/others = versão barata: só para o custo de associação
        # interno do DeepSORT dentro da mesma câmara.
        tracks = self.tracker.update_tracks(bbs, embeds=det_features_fast, others=det_features_fast)

        # 1. Identifica quais IDs locais do DeepSORT realmente estão válidos neste exato frame
        current_active_locals = {track.track_id for track in tracks if track.is_confirmed() and track.time_since_update == 0}

        # 2. Limpa o mapa de referências ANTES de buscar novos IDs globais
        lost_locals = [lid for lid in self.local_to_global_map if lid not in current_active_locals]
        for lid in lost_locals:
            del self.local_to_global_map[lid]

        # 3. Constrói a lista de IDs globais bloqueados usando apenas as pessoas realmente ativas
        active_global_ids = set(self.local_to_global_map.values())
        active_local_ids = current_active_locals

        do_feature_update = (self.frame_counter % self.feature_update_interval == 0)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            local_id = track.track_id
            active_local_ids.add(local_id)

            ltrb = track.to_ltrb()
            box = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]

            if local_id not in self.local_to_global_map:
                # Momento crítico (a): decisão cross-camera. Sempre com a
                # versão RICA, calculada agora, na hora, sobre esta caixa.
                rich_feature = extract_color_histogram_rich(frame, box)
                norm = np.linalg.norm(rich_feature)
                rich_feature = rich_feature / norm if norm > 0 else rich_feature

                global_id = self.global_manager.get_or_create_global_id(
                    new_feature_vector=rich_feature,
                    bbox=box,
                    cam_id=self.cam_id,
                    current_time=current_time,
                    active_global_ids=active_global_ids
                )
                self.local_to_global_map[local_id] = global_id
                active_global_ids.add(global_id)
            else:
                global_id = self.local_to_global_map[local_id]

                sent_feature = None
                if do_feature_update:
                    # Momento (b): atualização periódica do EMA global,
                    # também com a versão RICA -- mas só a cada N frames,
                    # então o custo extra é desprezível.
                    rich_feature = extract_color_histogram_rich(frame, box)
                    norm = np.linalg.norm(rich_feature)
                    sent_feature = rich_feature / norm if norm > 0 else rich_feature

                self.global_manager.update_existing_identity(
                    global_id=global_id,
                    new_feature_vector=sent_feature,
                    bbox=box,
                    cam_id=self.cam_id,
                    current_time=current_time
                )

            is_stopped, elapsed = self.state_tracker.update_and_evaluate(global_id, box, current_time)
            draw_person_annotation(frame, box, global_id, is_stopped, elapsed, self.config)

        cv2.rectangle(frame, (5, 5), (160, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        return frame

    def normalize_feature(feature_vector: np.ndarray) -> np.ndarray:
        """Garante que o vetor de embeddings tenha norma L2 = 1.0 para cálculo de cosseno."""
        norm = np.linalg.norm(feature_vector)
        if norm == 0:
            return feature_vector
        return (feature_vector / norm).astype(np.float32)

    def cleanup(self):
        print(f"[Worker-{self.cam_id}] Recursos libertados.")