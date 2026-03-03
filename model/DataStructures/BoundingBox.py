from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=False)
class BoundingBox:
    """
    Representa uma caixa delimitadora (ROI) com coordenadas e dimensões.
    """
    x: float
    y: float
    width: float
    height: float

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Retorna (x, y, w, h) como inteiros para uso no OpenCV."""
        return (int(self.x), int(self.y), int(self.width), int(self.height))

    def shift(self, delta_x: float, delta_y: float) -> 'BoundingBox':
        """
        Gera uma nova instância deslocada. 
        Útil para manter a integridade dos dados durante o rastreamento.
        """
        return BoundingBox(
            self.x + delta_x, 
            self.y + delta_y, 
            self.width, 
            self.height
        )