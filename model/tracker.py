# model/tracker.py
import cv2
import numpy as np
from model.data_structures import BoundingBox, TrackingData

class OpticalFlowTracker:
    def __init__(self, initial_frame_gray, initial_box_tuple):
        self._feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self._lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        initial_box = BoundingBox(*initial_box_tuple)
        initial_points = self._find_initial_points(initial_frame_gray, initial_box)
        
        self.tracking_data = TrackingData(
            previous_frame_gray=initial_frame_gray,
            points_to_track=initial_points,
            last_known_box=initial_box
        )

    def _find_initial_points(self, frame_gray, box):
        x, y, w, h = box.to_tuple()
        roi_gray = frame_gray[y:y+h, x:x+w]
        
        # Garante que a ROI não esteja vazia antes de procurar pontos
        if roi_gray.size == 0:
            return None
            
        points = cv2.goodFeaturesToTrack(roi_gray, mask=None, **self._feature_params)
        
        if points is not None:
            points[:, 0, 0] += x
            points[:, 0, 1] += y
        return points

    def update(self, new_frame_gray: np.ndarray):
        """
        Orquestra a atualização da posição do rastreador.
        Retorna (sucesso, lista_de_caixas).
        """
        if self.tracking_data.points_to_track is None:
            self.tracking_data.previous_frame_gray = new_frame_gray
            return (True, [self.tracking_data.last_known_box.to_tuple()])

        new_points, status = self._calculate_flow(new_frame_gray)

        # --- NOVA VERIFICAÇÃO DE SEGURANÇA ---
        if new_points is None:
            # Se o fluxo óptico falhar e não retornar pontos, paramos o processamento
            # deste frame e mantemos a última posição conhecida.
            self.tracking_data.previous_frame_gray = new_frame_gray
            return (True, [self.tracking_data.last_known_box.to_tuple()])
        # -------------------------------------

        good_new_points, good_old_points = self._filter_good_points(new_points, status)
        
        self._update_box_position(good_old_points, good_new_points)
        self._update_tracking_points(good_new_points)

        self.tracking_data.previous_frame_gray = new_frame_gray
        return (True, [self.tracking_data.last_known_box.to_tuple()])

    def _calculate_flow(self, new_frame_gray: np.ndarray):
        """Calcula o fluxo óptico e retorna os novos pontos e o status."""
        # Retorna None se não houver pontos de entrada para o cálculo
        if self.tracking_data.points_to_track is None or len(self.tracking_data.points_to_track) == 0:
            return None, None

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.tracking_data.previous_frame_gray, 
            new_frame_gray, 
            self.tracking_data.points_to_track, 
            None, 
            **self._lk_params
        )
        return new_points, status

    def _filter_good_points(self, new_points, status):
        """Filtra os pontos que foram rastreados com sucesso."""
        good_new = new_points[status == 1]
        good_old = self.tracking_data.points_to_track[status == 1]
        return good_new, good_old

    def _update_box_position(self, old_points, new_points):
        """Calcula o deslocamento e atualiza a posição da caixa."""
        if len(new_points) > 1: # Precisa de pelo menos 2 pontos para um deslocamento confiável
            delta_x = np.median([p_new[0] - p_old[0] for p_new, p_old in zip(new_points, old_points)])
            delta_y = np.median([p_new[1] - p_old[1] for p_new, p_old in zip(new_points, old_points)])
            
            self.tracking_data.last_known_box = self.tracking_data.last_known_box.shift(delta_x, delta_y)
            
    def _update_tracking_points(self, good_new_points):
        """Atualiza a lista de pontos para o próximo frame."""
        self.tracking_data.points_to_track = good_new_points.reshape(-1, 1, 2)