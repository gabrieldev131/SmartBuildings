import time
from typing import Optional

class StatefulTimer:
    """
    Encapsula a lógica de um temporizador que pode ser iniciado, 
    resetado e verificado para controle de permanência.
    """
    def __init__(self):
        self._start_time: Optional[float] = None

    def start_if_needed(self):
        """Inicia o temporizador se ele ainda não estiver rodando."""
        if self._start_time is None:
            self._start_time = time.time()

    def return_time_elapsed(self) -> float:
        """Retorna o tempo decorrido em segundos."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def reset(self):
        """Reseta o estado do temporizador."""
        self._start_time = None

    def has_exceeded(self, duration_seconds: float) -> bool:
        """Verifica se o tempo decorrido excedeu o limite."""
        if self._start_time is None:
            return False
        return self.return_time_elapsed() > duration_seconds
    
    def reload_timer(self, saved_time: float):
        """Ajusta o início do temporizador com base em um tempo já decorrido."""
        self._start_time = time.time() - saved_time