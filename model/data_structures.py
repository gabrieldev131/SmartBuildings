# model/data_structures.py

from dataclasses import dataclass
import numpy as np

import time

class StatefulTimer:
    """Encapsula a lógica de um temporizador que pode ser iniciado, resetado e verificado."""
    _start_time: float
    def __init__(self):
        try:
            if self._start_time >= 0.0:
                return
            
        except AttributeError:
            self._start_time = None

        except NameError:
            self._start_time = None 

    def start_if_needed(self):
        """Inicia o temporizador se ele ainda não estiver rodando."""
        if self._start_time == None:
            self._start_time = time.time()

    def return_time_elapsed(self) -> float:
        """Retorna o tempo decorrido desde o início do temporizador."""
        if self._start_time == None:
            return 0.0
        return time.time() - self._start_time
    
    def reset(self):
        """Reseta o temporizador."""
        self._start_time = None

    def has_exceeded(self, duration_seconds: float) -> bool:
        """Verifica se o tempo decorrido desde o início excedeu a duração especificada."""
        if self._start_time == None:
            return False
        return (time.time() - self._start_time) > duration_seconds
    
    def reload_timer(self, saved_time: float):
        """Recarrega o temporizador com um tempo salvo."""
        self._start_time = time.time() - saved_time
    
@dataclass
class BoundingBox:
    """Representa uma caixa delimitadora com coordenadas e dimensões."""
    x: float
    y: float
    width: float
    height: float

    def to_tuple(self):
        """Converte a caixa para uma tupla de inteiros (x, y, w, h)."""
        return (int(self.x), int(self.y), int(self.width), int(self.height))

    def shift(self, delta_x: float, delta_y: float):
        """Retorna uma nova BoundingBox deslocada, promovendo a imutabilidade."""
        return BoundingBox(self.x + delta_x, self.y + delta_y, self.width, self.height)

@dataclass
class TrackingData:
    """Agrupa todos os dados de estado de um rastreador."""
    previous_frame_gray: np.ndarray
    points_to_track: np.ndarray
    last_known_box: BoundingBox