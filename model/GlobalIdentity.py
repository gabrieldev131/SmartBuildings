# model/GlobalIdentity.py
import numpy as np
import time
from datetime import datetime

class GlobalIdentity:
    """
    TAD que representa uma pessoa rastreada no sistema multi-câmara.
    """
    def __init__(self, global_id: int, initial_feature: np.ndarray, initial_cam_id: str, start_time: float):
        self.global_id = global_id
        self.feature_vector = initial_feature
        
        self.first_seen = start_time
        self.last_seen = start_time
        self.current_camera = initial_cam_id
        
        self.camera_history = [[initial_cam_id, start_time, None]]

    def update_location(self, cam_id: str, current_time: float) -> bool:
        """
        Atualiza a posição da pessoa e gere o histórico.
        Retorna True se houve uma mudança de câmara (Transição), False caso contrário.
        """
        self.last_seen = current_time
        
        if self.current_camera != cam_id:
            if self.camera_history:
                self.camera_history[-1][2] = current_time
            
            self.current_camera = cam_id
            self.camera_history.append([cam_id, current_time, None])
            return True 
            
        return False 

    def get_total_duration(self) -> float:
        return self.last_seen - self.first_seen

    # -------------------------------------------------------------------------
    # MÉTODOS DE EXPORTAÇÃO PARA ESTATÍSTICA (NOVO)
    # -------------------------------------------------------------------------

    def get_raw_history(self) -> list[dict]:
        """
        Retorna o histórico com dados brutos (float) ideais para análise de dados,
        Cadeias de Markov e Monte Carlo.
        """
        raw_data = []
        for record in self.camera_history:
            cam_id, t_in, t_out = record
            
            # Se a pessoa ainda está na câmara no momento da exportação, fechamos o tempo com o tempo atual
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

    # -------------------------------------------------------------------------
    # MÉTODOS DE FORMATAÇÃO (CAMADA DE APRESENTAÇÃO)
    # -------------------------------------------------------------------------

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
        
    def print_summary(self):
        duration = self.get_total_duration()
        print(f"\n--- Resumo do ID Global {self.global_id} ---")
        print(f"Tempo total no prédio: {duration:.1f} segundos")
        print("Rota percorrida:")
        for step in self.get_human_readable_history():
            print(f" -> {step['Camera']} | In: {step['Entrada']} | Out: {step['Saida']}")
        print("---------------------------------\n")