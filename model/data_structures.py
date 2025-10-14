# model/data_structures.py

from dataclasses import dataclass
import numpy as np

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