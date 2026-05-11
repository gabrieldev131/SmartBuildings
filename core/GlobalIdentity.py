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
        self.current_camera = initial_cam_id
        self.last_seen_per_camera = {initial_cam_id: start_time}
        self.camera_history = [[initial_cam_id, start_time, None]]

    def update(self, new_feature_vector: np.ndarray, bbox: list, cam_id: str, current_time: float, switch_cooldown: float = 2.0) -> bool:
        
        if new_feature_vector is not None:
            # MUDANÇA CRUCIAL: EMA mais lento. 
            # A memória pesa 90%, a cor nova apenas 10%. Assim evitamos esquecer a cor real 
            # se a pessoa passar momentaneamente por uma sombra intensa.
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

    def get_total_duration(self) -> float:
        return self.last_seen - self.first_seen

    def get_raw_history(self) -> list[dict]:
        raw_data = []
        for record in self.camera_history:
            cam_id, t_in, t_out = record
            if t_out is None:
                t_out = time.time()
            raw_data.append({
                "global_id": self.global_id,
                "camera_id": cam_id,
                "timestamp_in": t_in,
                "timestamp_out": t_out,
                "duration_seconds": t_out - t_in
            })
        return raw_data

    def _format_timestamp(self, timestamp: float) -> str:
        if timestamp is None:
            return "Ainda na câmara"
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object.strftime("%H:%M:%S:%d/%m/%Y")

    def get_human_readable_history(self) -> list[dict]:
        formatted_history = []
        for record in self.camera_history:
            cam_id, t_in, t_out = record
            formatted_history.append({
                "Camera": cam_id,
                "Entrada": self._format_timestamp(t_in),
                "Saida": self._format_timestamp(t_out)
            })
        return formatted_history