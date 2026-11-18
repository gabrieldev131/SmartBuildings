# domain/AnnotationSpec.py
from dataclasses import dataclass


@dataclass(slots=True)
class AnnotationSpec:
    """Visual specification for rendering a tracked person on the HUD."""
    bbox: list[float]
    global_id: int
    label: str
    color: tuple[int, int, int]
