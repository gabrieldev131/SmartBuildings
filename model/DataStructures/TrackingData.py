from dataclasses import dataclass
import numpy as np
from model.DataStructures.BoundingBox import BoundingBox

@dataclass
class TrackingData:
    """
    Estrutura de dados para persistência ou transferência de estado 
    entre ciclos de processamento.
    """
    previous_frame_gray: np.ndarray
    points_to_track: np.ndarray
    last_known_box: BoundingBox