# domain/TrackContext.py
from dataclasses import dataclass


@dataclass(slots=True)
class TrackContext:
    """Context passed to each `IBehaviorAnalyzer` plugin on every frame for a tracked person."""
    global_id: int
    local_id: int
    cam_id: str
    bbox: list[float]
    current_time: float
    confidence: float = 1.0
