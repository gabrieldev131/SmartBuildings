# controller/camera_processor.py
import cv2
import time
import threading
import logging
import numpy as np
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from model.ProcessingWorker import detect_people_in_frame


class CameraProcessor:
    def __init__(self, source_id, config, detection_lock: threading.Lock, global_id_manager=None):
        self.source_id = source_id
        self.config = config
        self.frame_counter = 0

        # CORRECAO: Lock recebido do AppController para proteger pending_detections.
        # O callback do pool.apply_async roda em uma thread auxiliar do processo
        # principal, enquanto process_frame() roda na thread principal do loop.
        # Sem o lock, ambas podem ler/escrever pending_detections ao mesmo tempo.
        self._detection_lock = detection_lock
        self._pending_detections = None
        self._frame_for_tracker = None

        self.global_id_manager = global_id_manager
        self.local_to_global_map = {}
        # --- Parametros do DeepSORT ---
        # max_age: quantos frames o tracker sobrevive SEM ser associado a uma deteccao.
        #   Regra: deve ser BEM maior que DETECT_EVERY_N_FRAMES para aguentar
        #   multiplos ciclos sem deteccao. Ex: DETECT_EVERY_N_FRAMES=30 -> max_age >= 90.
        #   Com 90, a pessoa pode ficar 3 ciclos invisiveis e ainda manter o ID.
        #
        # n_init: quantas deteccoes consecutivas sao necessarias para CONFIRMAR um track.
        #   Com n_init=1 qualquer deteccao espuria vira um ID imediatamente.
        #   Com n_init=3 o tracker precisa ver a pessoa em 3 frames seguidos antes
        #   de exibir o ID, eliminando falsos positivos de curta duracao.
        #
        # max_cosine_distance: limiar de similaridade de aparencia (ReID).
        #   Valores menores = mais exigente para reutilizar um ID antigo.
        #   0.3-0.4 e um bom balanco para cameras fixas com imagem boa.
        #
        # embedder_wts: usa MobileNet local para extrair features de aparencia,
        #   tornando a re-identificacao mais robusta que so posicao.
        self.tracker = DeepSort(
            max_age=90,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.6,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
        self.current_tracks = []

        # Estado de parada: usa deque de historico por ID para ser mais robusto
        # do que comparar apenas dois frames consecutivos.
        # Formato: { track_id: deque([(timestamp, (cx, cy)), ...]) }
        self._position_history: dict[int, deque] = {}

        # Cache de estado de parada para nao recalcular a cada frame
        self._stopped_state: dict[int, bool] = {}
        self._stopped_since: dict[int, float] = {}

        # Metrica: conta frames descartados por fila cheia no worker
        self._dropped_detection_frames = 0

    # -------------------------------------------------------------------------
    # INTERFACE PUBLICA
    # -------------------------------------------------------------------------

    def process_frame(self, frame, detection_pool):
        self.frame_counter += 1

        if self._is_detection_frame() and self.config.USE_MULTIPROCESSING and detection_pool:
            # CORREÇÃO TEMPORAL: Guardamos o frame EXATO em que a pessoa se encontra agora
            self._frame_for_tracker = frame.copy() 
            
            detection_pool.apply_async(
                detect_people_in_frame,
                args=(frame, self.config.YOLO_CONFIDENCE_THRESHOLD, self.config.YOLO_NMS_THRESHOLD),
                callback=self._on_detection_result,
                error_callback=self._on_detection_error
            )

        with self._detection_lock:
            if self._pending_detections is not None:
                # Usa o frame guardado para extrair as características corretamente.
                # Se por algum motivo não existir, usa o atual como fallback de segurança.
                track_frame = self._frame_for_tracker if self._frame_for_tracker is not None else frame
                
                self.current_tracks = self.tracker.update_tracks(
                    self._pending_detections, frame=track_frame
                )
                
                # Limpa os estados após a atualização
                self._pending_detections = None
                self._frame_for_tracker = None

        boxes_and_states = self._build_boxes_and_states()
        processed_frame = self._draw_boxes(frame, boxes_and_states)
        self._cleanup_old_states([t.track_id for t in self.current_tracks if t.is_confirmed()])

        return processed_frame

    # -------------------------------------------------------------------------
    # CALLBACKS DO POOL (rodam em thread auxiliar do processo principal)
    # -------------------------------------------------------------------------

    def _on_detection_result(self, yolo_results):
        """
        Callback chamado quando o processo worker termina a deteccao YOLO.
        CORRECAO: Protegido por lock pois roda em thread diferente do loop principal.
        """
        formatted = [(box, conf, 'person') for box, conf in yolo_results]
        with self._detection_lock:
            self._pending_detections = formatted

    def _on_detection_error(self, exc):
        """Callback de erro do pool — loga sem travar o sistema."""
        logging.error(f"[{self.source_id}] Erro no worker de deteccao: {exc}")

    # -------------------------------------------------------------------------
    # LOGICA DE ESTADO DE PARADA (com janela de tempo — mais robusto)
    # -------------------------------------------------------------------------

    def _build_boxes_and_states(self):
        """Itera sobre tracks confirmados e calcula estado de parada com IDs Globais."""
        boxes_and_states = []
        current_time = time.time()

        for track in self.current_tracks:
            if not track.is_confirmed() or track.time_since_update > 15:
                continue

            local_id = track.track_id
            ltrb = track.to_ltrb()
            x, y = int(ltrb[0]), int(ltrb[1])
            w, h = int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1])
            box = [x, y, w, h]

            # --- LÓGICA DE RE-IDENTIFICAÇÃO ---
            display_id = local_id # Fallback de segurança

            if self.global_id_manager is not None:
                if local_id not in self.local_to_global_map:
                    if track.features:
                        latest_feature = track.features[-1]
                        global_id = self.global_id_manager.get_or_create_global_id(latest_feature)
                        self.local_to_global_map[local_id] = global_id
                        display_id = global_id
                    else:
                        # CORREÇÃO: Usamos o ID local apenas para mostrar na tela neste frame, 
                        # mas NÃO o guardamos no dicionário! Assim, no próximo frame 
                        # ele volta a tentar extrair e enviar para o Gestor Global.
                        display_id = local_id 
                else:
                    # Se já foi mapeado com sucesso antes, usa a tradução
                    display_id = self.local_to_global_map[local_id]
            else:
                display_id = local_id # Caso o gestor não exista

            # --- ATUALIZAR ESTADOS COM O ID GLOBAL ---
            self._update_position_history(display_id, box, current_time)
            is_stopped = self._evaluate_stopped_state(display_id, current_time)
            elapsed = self._get_stopped_elapsed(display_id, current_time)

            boxes_and_states.append((box, is_stopped, display_id, elapsed))

        return boxes_and_states

    def _update_position_history(self, track_id, box, current_time):
        """Adiciona a posicao atual ao historico deslizante do ID."""
        if track_id not in self._position_history:
            self._position_history[track_id] = deque(maxlen=90)  # ~3s a 30fps
            self._stopped_state[track_id] = False
            self._stopped_since[track_id] = current_time

        x, y, w, h = box
        center = (x + w / 2, y + h / 2)
        self._position_history[track_id].append((current_time, center))

    def _evaluate_stopped_state(self, track_id, current_time) -> bool:
        """
        Avalia se a pessoa esta parada com base no deslocamento dentro
        de uma janela de tempo (mais resistente a jitter do DeepSORT).
        """
        history = self._position_history.get(track_id)
        if not history or len(history) < 2:
            return False

        window_sec = self.config.STOPPED_SECONDS_THRESHOLD
        recent = [pt for pt in history if current_time - pt[0] <= window_sec]

        if len(recent) < 2:
            return False

        start_pos = recent[0][1]
        end_pos = recent[-1][1]
        displacement = np.sqrt(
            (end_pos[0] - start_pos[0]) ** 2 +
            (end_pos[1] - start_pos[1]) ** 2
        )

        currently_stopped = self._stopped_state[track_id]

        if currently_stopped:
            # So sai do estado parado com movimento significativo (histerese)
            if displacement > self.config.MOVEMENT_BREAKOUT_THRESHOLD:
                self._stopped_state[track_id] = False
                self._stopped_since[track_id] = current_time
        else:
            if displacement < self.config.STOPPED_PIXEL_THRESHOLD:
                # Acumula tempo parado
                time_still = current_time - self._stopped_since[track_id]
                if time_still >= self.config.STOPPED_SECONDS_THRESHOLD:
                    self._stopped_state[track_id] = True
            else:
                # Resetar o cronometro a cada movimento detectado
                self._stopped_since[track_id] = current_time

        return self._stopped_state[track_id]

    def _get_stopped_elapsed(self, track_id, current_time) -> float:
        """Retorna quanto tempo (s) a pessoa esta parada. Zero se estiver movendo."""
        if not self._stopped_state.get(track_id, False):
            return 0.0
        return current_time - self._stopped_since.get(track_id, current_time)

    # -------------------------------------------------------------------------
    # DESENHO
    # -------------------------------------------------------------------------

    def _draw_boxes(self, frame, boxes_and_states):
        for box, is_stopped, track_id, elapsed in boxes_and_states:
            color = self.config.STOPPED_BOX_COLOR if is_stopped else self.config.PERSON_BOX_COLOR
            x, y, w, h = [int(v) for v in box]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"ID:{track_id}"
            if is_stopped:
                label += f" PARADO {elapsed:.0f}s"

            # Fundo escuro no texto para melhor leitura
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - th - 6), (x + tw + 2, y), (0, 0, 0), -1)
            cv2.putText(frame, label, (x + 1, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return frame

    # -------------------------------------------------------------------------
    # LIMPEZA DE MEMORIA
    # -------------------------------------------------------------------------

    def _is_detection_frame(self):
        return self.frame_counter % self.config.DETECT_EVERY_N_FRAMES == 0

    def _cleanup_old_states(self, active_track_ids):
        """Remove histórico de IDs Globais que saíram de cena."""
        # active_track_ids contém os IDs locais do DeepSORT que ainda estão visíveis.
        
        # Descobre quais IDs locais desapareceram
        inactive_locals = [tid for tid in self.local_to_global_map if tid not in active_track_ids]
        
        for local_tid in inactive_locals:
            # Remove o ID local do mapa de tradução e recupera o ID Global
            global_tid = self.local_to_global_map.pop(local_tid, None)
            
            # Limpa o histórico de posições usando o ID Global
            if global_tid is not None:
                self._position_history.pop(global_tid, None)
                self._stopped_state.pop(global_tid, None)
                self._stopped_since.pop(global_tid, None)