# core/GlobalIdentity.py
import numpy as np
import time
from datetime import datetime

class GlobalIdentity:
    def __init__(self, global_id: int, initial_feature: np.ndarray, bbox: list, initial_cam_id: str, start_time: float):
        self.global_id = global_id
        self.feature_vector = initial_feature
        self.last_bbox = bbox
        
        self.first_seen = start_time
        self.last_seen = start_time
        
        # Mantemos o registo exato de câmaras. A abstração de cluster
        # será calculada dinamicamente pelo Manager.
        self.current_camera = initial_cam_id
        self.last_seen_per_camera = {initial_cam_id: start_time}
        self.camera_history = [[initial_cam_id, start_time, None]]

    def update(self, new_feature_vector: np.ndarray, bbox: list, cam_id: str, current_time: float, switch_cooldown: float = 2.0) -> bool:
        if new_feature_vector is not None:
            self.feature_vector = 0.9 * self.feature_vector + 0.1 * new_feature_vector
            
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
        return False 

    def get_raw_history(self) -> list[dict]:
        """Retorna o histórico bruto. O Manager é quem o transformará em histórico de ambientes."""
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