# domain/BehaviorResult.py
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class BehaviorResult:
    """Standardized output returned by an `IBehaviorAnalyzer` plugin."""
    name: str
    is_active: bool = False
    label_suffix: str = ""
    color_override: Optional[tuple[int, int, int]] = None
    event_type: Optional[str] = None
    duration_seconds: float = 0.0
    details: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)
