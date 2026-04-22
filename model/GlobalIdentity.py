# model/GlobalIdentity.py
import numpy as np
from datetime import datetime

class GlobalIdentity:
    """
    TAD que representa uma pessoa rastreada no sistema multi-câmara.
    """
    def __init__(self, global_id: int, initial_feature: np.ndarray, initial_cam_id: str, start_time: float):
        self.global_id = global_id
        self.feature_vector = initial_feature
        
        # Mantemos os floats internamente para contas matemáticas rápidas
        self.first_seen = start_time
        self.last_seen = start_time
        self.current_camera = initial_cam_id
        
        self.camera_history = [[initial_cam_id, start_time, None]]

    def update_location(self, cam_id: str, current_time: float):
        """Atualiza a posição da pessoa e gere o histórico."""
        self.last_seen = current_time
        
        if self.current_camera != cam_id:
            if self.camera_history:
                self.camera_history[-1][2] = current_time
            
            self.current_camera = cam_id
            self.camera_history.append([cam_id, current_time, None])
            
    def get_total_duration(self) -> float:
        """Retorna o tempo total (em segundos)."""
        return self.last_seen - self.first_seen

    # -------------------------------------------------------------------------
    # NOVOS MÉTODOS DE FORMATAÇÃO (CAMADA DE APRESENTAÇÃO)
    # -------------------------------------------------------------------------

    def _format_timestamp(self, timestamp: float) -> str:
        """
        Método utilitário privado. 
        Converte o float bruto para o formato HH:MM:SS:DD/MM/YYYY.
        """
        if timestamp is None:
            return "Ainda na câmara"
        
        # Converte o float para um objeto de data/hora e formata como string
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object.strftime("%H:%M:%S:%d/%m/%Y")

    def get_human_readable_history(self) -> list[dict]:
        """
        Retorna o histórico de rotas num formato descritivo e fácil de ler.
        Ideal para logs, relatórios ou exportação para JSON/CSV.
        """
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
        """Imprime um resumo amigável no terminal."""
        duration = self.get_total_duration()
        print(f"--- Resumo do ID Global {self.global_id} ---")
        print(f"Tempo total no prédio: {duration:.1f} segundos")
        print("Rota percorrida:")
        for step in self.get_human_readable_history():
            print(f" -> {step['Camera']} | In: {step['Entrada']} | Out: {step['Saida']}")
        print("---------------------------------")