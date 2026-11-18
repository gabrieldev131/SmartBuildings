# domain/BoundingBox.py
from dataclasses import dataclass


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
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_ltwh(self) -> list[float]:
        return [self.x1, self.y1, self.width, self.height]
