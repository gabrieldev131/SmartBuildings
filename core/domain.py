# core/domain.py
"""
Domain Entities and Value Objects for the SmartBuildings Clean Architecture.
Pure Python/NumPy data structures with zero external framework dependencies.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(slots=True)
class BoundingBox:
    """Value Object representing a 2D bounding box in [x1, y1, x2, y2] format."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return self.width / h if h > 0 else 0.0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_ltwh(self) -> list[float]:
        return [self.x1, self.y1, self.width, self.height]


@dataclass(slots=True)
class FramePacket:
    """Value Object encapsulating a decoded video frame and its provenance metadata."""
    cam_id: str
    frame: np.ndarray
    timestamp: float
    seq_id: int = 0


@dataclass(slots=True)
class TrackContext:
    """
    Context passed to each `IBehaviorAnalyzer` plugin on every frame for a tracked person.
    """
    global_id: int
    local_id: int
    cam_id: str
    bbox: list[float]          # [x1, y1, x2, y2]
    current_time: float
    confidence: float = 1.0


@dataclass(slots=True)
class BehaviorResult:
    """
    Standardized output returned by an `IBehaviorAnalyzer` plugin.
    Allows new functionalities to control HUD labels, box colors, and CSV event logging
    without modifying `CameraWorker`, `display.py`, or `GlobalIdentityManager`.
    """
    name: str                                          # Unique analyzer key (e.g., "stopped", "zone_alert")
    is_active: bool = False                            # Whether the behavior/alert is currently active
    label_suffix: str = ""                             # Text appended to the person's HUD label (e.g., "PARADO 5s")
    color_override: Optional[tuple[int, int, int]] = None  # BGR box color when active (e.g., (0, 0, 255))
    event_type: Optional[str] = None                   # CSV event_type when active (e.g., "PERSON_STOPPED")
    duration_seconds: float = 0.0                      # Duration of the active behavior/episode
    details: str = ""                                  # Human-readable description for CSV export


@dataclass(slots=True)
class AnnotationSpec:
    """Visual specification for rendering a tracked person on the HUD."""
    bbox: list[float]
    global_id: int
    label: str
    color: tuple[int, int, int]
