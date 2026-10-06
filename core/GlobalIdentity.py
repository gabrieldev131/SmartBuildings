# core/GlobalIdentity.py
import numpy as np
import time

class GlobalIdentity:
    def __init__(self, global_id: int, initial_feature: np.ndarray, bbox: list, initial_cam_id: str, start_time: float, max_gallery_size: int = 6):
        self.global_id = global_id
        
        norm = np.linalg.norm(initial_feature)
        unit_feat = initial_feature / norm if norm > 0 else initial_feature
        
        # Galeria mantem os ultimos N vetores validos
        self.feature_gallery = [unit_feat]
        self.max_gallery_size = max_gallery_size
        
        self.last_bbox = bbox
        self.first_seen = start_time
        self.last_seen = start_time
        
        self.current_camera = initial_cam_id
        self.last_seen_per_camera = {initial_cam_id: start_time}
        self.camera_history = [[initial_cam_id, start_time, None]]

    @property
    def feature_vector(self) -> np.ndarray:
        # Vetor medio da galeria para exportacao/referencia
        mean_feat = np.mean(self.feature_gallery, axis=0)
        norm = np.linalg.norm(mean_feat)
        return mean_feat / norm if norm > 0 else mean_feat

    def match_score(self, candidate_feature: np.ndarray) -> float:
        """
        Retorna a menor distancia de aparencia em relacao a TODOS os vetores da galeria.
        Garante associacao mesmo com variacoes de angulo ou postura.
        """
        norm = np.linalg.norm(candidate_feature)
        candidate_norm = candidate_feature / norm if norm > 0 else candidate_feature
        
        dots = np.dot(self.feature_gallery, candidate_norm)
        dots = np.clip(dots, -1.0, 1.0)
        best_similarity = np.max(dots)
        return float(max(0.0, 1.0 - best_similarity))

    def update(self, new_feature_vector: np.ndarray, bbox: list, cam_id: str, current_time: float, switch_cooldown: float = 2.0) -> bool:
        if new_feature_vector is not None:
            norm = np.linalg.norm(new_feature_vector)
            if norm > 0:
                unit_feat = new_feature_vector / norm
                
                # Adiciona à galeria se for minimamente consistente
                self.feature_gallery.append(unit_feat)
                if len(self.feature_gallery) > self.max_gallery_size:
                    self.feature_gallery.pop(0)

        self.last_bbox = bbox
        self.last_seen = current_time
        self.last_seen_per_camera[cam_id] = current_time
        
        if self.current_camera != cam_id:
            time_since_primary = current_time - self.last_seen_per_camera.get(self.current_camera, 0)
            if time_since_primary > switch_cooldown:
                time_of_exit = self.last_seen_per_camera[self.current_camera]
                if self.camera_history:
                    self.camera_history[-1][2] = time_of_exit
                self.current_camera = cam_id
                self.camera_history.append([cam_id, time_of_exit, None])
                return True
        return False

    def get_raw_history(self) -> list[dict]:
        raw_data = []
        for record in self.camera_history:
            cam_id, t_in, t_out = record
            if t_out is None:
                t_out = time.time()
            raw_data.append({
                "camera_id": cam_id,
                "timestamp_in": t_in,
                "timestamp_out": t_out
            })
        return raw_data