# domain/FramePacket.py
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class FramePacket:
    """Value Object encapsulating a decoded video frame and its provenance metadata."""
    cam_id: str
    frame: np.ndarray
    timestamp: float
    seq_id: int = 0
