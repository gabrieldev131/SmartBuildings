# behaviors/RestrictedZoneAnalyzer.py
from behaviors.BehaviorRegistry import BehaviorRegistry
from domain.BehaviorResult import BehaviorResult
from domain.TrackContext import TrackContext
from interfaces.IBehaviorAnalyzer import IBehaviorAnalyzer


@BehaviorRegistry.register("restricted_zone")
class RestrictedZoneAnalyzer(IBehaviorAnalyzer):
    """Alerts when a tracked person enters a rectangular normalized zone."""

    def __init__(self, config, zone_norm: tuple[float, float, float, float] = (0.0, 0.0, 0.25, 0.25)):
        w = getattr(config, "PROCESSING_WIDTH", 640)
        h = getattr(config, "PROCESSING_HEIGHT", 480)
        self.x1, self.y1 = zone_norm[0] * w, zone_norm[1] * h
        self.x2, self.y2 = zone_norm[2] * w, zone_norm[3] * h
        self.entered_since: dict[tuple[int, str], float] = {}

    @classmethod
    def from_config(cls, config) -> "RestrictedZoneAnalyzer":
        zone_rect = getattr(config, "RESTRICTED_ZONE_NORM", (0.0, 0.0, 0.25, 0.25))
        return cls(config, zone_norm=zone_rect)

    def _is_inside(self, bbox: list[float]) -> bool:
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        return (self.x1 <= cx <= self.x2) and (self.y1 <= cy <= self.y2)

    def analyze(self, context: TrackContext) -> BehaviorResult:
        key = (context.global_id, context.cam_id)
        if not self._is_inside(context.bbox):
            self.entered_since.pop(key, None)
            return BehaviorResult(name="restricted_zone", is_active=False, event_type="ZONE_INTRUSION")

        self.entered_since.setdefault(key, context.current_time)
        elapsed = context.current_time - self.entered_since[key]
        return BehaviorResult(
            name="restricted_zone",
            is_active=True,
            label_suffix="ZONA RESTRITA",
            color_override=(0, 165, 255),
            event_type="ZONE_INTRUSION",
            duration_seconds=elapsed,
            details=f"Pessoa ID {context.global_id} na zona restrita em '{context.cam_id}' ({elapsed:.1f}s)",
        )
