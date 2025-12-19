# model/tracker.py
import cv2
import numpy as np
import time
from collections import deque
from model.data_structures import BoundingBox, StatefulTimer
import config

class StatefulTracker:
    _next_id = 0

    def __init__(self, initial_frame_gray, initial_box_tuple):
        self.id = StatefulTracker._next_id
        StatefulTracker._next_id += 1

        self._feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self._lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.box = BoundingBox(*initial_box_tuple)
        self.history = deque(maxlen=100)
        self.last_gray_frame = initial_frame_gray
        self.points_to_track = self._find_initial_points(initial_frame_gray, self.box)
        
        self.is_stopped = False
        self.stopped_timer = StatefulTimer()

    def _find_initial_points(self, frame_gray, box):
        
        x, y, w, h = box.to_tuple()
        roi_gray = frame_gray[y:y+h, x:x+w]
        if roi_gray.size == 0: return None
        points = cv2.goodFeaturesToTrack(roi_gray, mask=None, **self._feature_params)
        if points is not None:
            points[:, 0, 0] += x
            points[:, 0, 1] += y
        return points
    
    def get_centroid(self):
        return (self.box.x + self.box.width / 2, self.box.y + self.box.height / 2)

    def update(self, new_frame_gray, stop_thresh_px, breakout_thresh_px, stop_thresh_sec, smoothing_factor):
        # ... (código de cálculo do fluxo óptico e atualização da caixa continua igual) ...
        if self.points_to_track is None or len(self.points_to_track) == 0:
            self.last_gray_frame = new_frame_gray
            return (True, self.box.to_tuple(), self.is_stopped)

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.last_gray_frame, new_frame_gray, self.points_to_track, None, **self._lk_params)
        if new_points is None:
            self.last_gray_frame = new_frame_gray
            return (True, self.box.to_tuple(), self.is_stopped)

        good_new = new_points[status == 1]
        good_old = self.points_to_track[status == 1]

        if len(good_new) > 1:
            dx = np.median([p_new[0] - p_old[0] for p_new, p_old in zip(good_new, good_old)])
            dy = np.median([p_new[1] - p_old[1] for p_new, p_old in zip(good_new, good_old)])
            calculated_box = self.box.shift(dx, dy)

            # --- LÓGICA DE SUAVIZAÇÃO DA CAIXA ---
            # Em vez de pular para a nova posição, nos movemos suavemente em direção a ela.
            current_x, current_y = self.box.x, self.box.y
            target_x, target_y = calculated_box.x, calculated_box.y

            new_x = (current_x * smoothing_factor) + (target_x * (1 - smoothing_factor))
            new_y = (current_y * smoothing_factor) + (target_y * (1 - smoothing_factor))
            
            # Atualiza a caixa com a nova posição suavizada
            self.box = BoundingBox(new_x, new_y, self.box.width, self.box.height)

        self.points_to_track = good_new.reshape(-1, 1, 2)
        self.history.append((time.time(), self.get_centroid()))
        self.last_gray_frame = new_frame_gray
        
        self._check_if_stopped(stop_thresh_px, breakout_thresh_px, stop_thresh_sec)


        # "Debugging"
        distance = self._calculate_movement_distance()
        print(f"Tracker {self.id} - Distance moved: {distance} pixels - Time: {self.stopped_timer.return_time_elapsed()}.")




        return (True, self.box.to_tuple(), self.is_stopped)

    def _check_if_stopped(self, stop_threshold, breakout_threshold, seconds_threshold):
        """
        Orquestra a lógica de verificação de parada, delegando para métodos de estado.
        """
        distance = self._calculate_movement_distance()
        if distance == None:
            return

        if self.is_stopped:
            self._process_stopped_state(distance, breakout_threshold)
            return

        self._process_moving_state(distance, stop_threshold, seconds_threshold)

        #"Debugging"
        if self.is_stopped:
            print(f"Tracker {self.id} - is stoped: {self.is_stopped}.")

#--------------------------------------------------------------------------------------------------------
    # --- NOVOS MÉTODOS PRIVADOS ---
    def _calculate_movement_distance(self):
        """Calcula a distância percorrida na janela de tempo do histórico."""
        current_time = time.time()
        time_window = 2.0  
        
        recent_history = [item for item in self.history if current_time - item[0] <= time_window]

        if len(recent_history) < 2:
            return None
        start_point = recent_history[0][1]
        end_point = recent_history[-1][1]
        return np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)

    def _process_stopped_state(self, distance, breakout_threshold):
        """Lida com a lógica quando o estado atual é 'parado'."""
        is_moving_significantly = distance > breakout_threshold
        if is_moving_significantly:
            self._transition_to_moving()

    def _process_moving_state(self, distance, stop_threshold, seconds_threshold):
        """Lida com a lógica quando o estado atual é 'movendo'."""
        if distance == None:
            return
        is_moving_slowly = distance < stop_threshold
        if not is_moving_slowly:
            self.stopped_timer.reset()
            return

        self.stopped_timer.start_if_needed()
        if self.stopped_timer.has_exceeded(seconds_threshold):
            self._transition_to_stopped()

    def _transition_to_moving(self):
        """Muda o estado para 'movendo' e reseta o timer."""
        self.is_stopped = False
        self.stopped_timer.reset()

    def _transition_to_stopped(self):
        """Muda o estado para 'parado'."""
        self.is_stopped = True
        config.TEST = False
    
    def get_time(self):
        return self.stopped_timer
    def get_state(self):
        return self.is_stopped
    

    def reload_time(self, time_elapsed):
        self.stopped_timer = time_elapsed
    def reload_state(self, state):
        self.is_stopped = state