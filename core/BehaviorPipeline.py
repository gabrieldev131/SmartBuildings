# core/BehaviorPipeline.py
"""
Decoupled Behavior Analysis Pipeline (Strategy + Composite + Observer Design Patterns).

How to add a new functionality (Open/Closed Principle - OCP):
1. Create a class implementing `IBehaviorAnalyzer` with `analyze(context: TrackContext) -> BehaviorResult`.
2. Add your class to `BehaviorPipeline.default_analyzers(config)` (or pass it to `BehaviorPipeline`).
3. You DO NOT need to edit `CameraWorker`, `ui/display.py`, or `GlobalIdentityManager` —
   HUD labels, box colors, and CSV event logging work automatically!
"""
from typing import Optional
from core.domain import AnnotationSpec, BehaviorResult, TrackContext
from core.ports import IBehaviorAnalyzer
from core.StoppedStateTracker import StoppedStateTracker


class RestrictedZoneAnalyzer(IBehaviorAnalyzer):
    """
    Example plug-and-play functionality implementing `IBehaviorAnalyzer`:
    Triggers an alert whenever a person's center enters a configured normalized zone
    `(x1_norm, y1_norm, x2_norm, y2_norm)` in `[0.0, 1.0]`.
    """

    def __init__(self, config, zone_norm: tuple[float, float, float, float] = (0.0, 0.0, 0.25, 0.25)):
        self.width = getattr(config, "PROCESSING_WIDTH", 640)
        self.height = getattr(config, "PROCESSING_HEIGHT", 480)
        self.x1 = zone_norm[0] * self.width
        self.y1 = zone_norm[1] * self.height
        self.x2 = zone_norm[2] * self.width
        self.y2 = zone_norm[3] * self.height
        self.entered_since: dict[tuple[int, str], float] = {}

    def analyze(self, context: TrackContext) -> BehaviorResult:
        cx = (context.bbox[0] + context.bbox[2]) * 0.5
        cy = (context.bbox[1] + context.bbox[3]) * 0.5
        inside = (self.x1 <= cx <= self.x2) and (self.y1 <= cy <= self.y2)
        key = (context.global_id, context.cam_id)

        if inside:
            if key not in self.entered_since:
                self.entered_since[key] = context.current_time
            elapsed = context.current_time - self.entered_since[key]
            return BehaviorResult(
                name="restricted_zone",
                is_active=True,
                label_suffix="ZONA RESTRITA",
                color_override=(0, 165, 255),  # Orange alert box in BGR
                event_type="ZONE_INTRUSION",
                duration_seconds=elapsed,
                details=f"Pessoa ID {context.global_id} entrou na zona restrita em '{context.cam_id}' ({elapsed:.1f}s)",
            )

        self.entered_since.pop(key, None)
        return BehaviorResult(name="restricted_zone", is_active=False, event_type="ZONE_INTRUSION")


class BehaviorPipeline:
    """
    Composite pipeline that executes all registered `IBehaviorAnalyzer` plugins for each person,
    aggregates their visual annotations, and dispatches events to `GlobalIdentityManager`.
    """

    def __init__(self, config, analyzers: Optional[list[IBehaviorAnalyzer]] = None):
        self.config = config
        self.default_color = getattr(config, "COLOR_MOVING", (0, 255, 0))
        self.analyzers: list[IBehaviorAnalyzer] = (
            list(analyzers) if analyzers is not None else self.default_analyzers(config)
        )
        self._prev_active_state: dict[tuple[int, str], bool] = {}

    @staticmethod
    def default_analyzers(config) -> list[IBehaviorAnalyzer]:
        """
        Factory method returning the active behavior plugins based on `Config`.
        Add new `IBehaviorAnalyzer` instances here to extend the system!
        """
        plugins: list[IBehaviorAnalyzer] = []
        if getattr(config, "ENABLE_STOPPED_DETECTION", True):
            plugins.append(StoppedStateTracker(config))
        if getattr(config, "ENABLE_RESTRICTED_ZONE", False):
            zone_rect = getattr(config, "RESTRICTED_ZONE_NORM", (0.0, 0.0, 0.25, 0.25))
            plugins.append(RestrictedZoneAnalyzer(config, zone_norm=zone_rect))
        return plugins

    def register(self, analyzer: IBehaviorAnalyzer) -> None:
        """Dynamically attach a new behavior analyzer plugin at runtime."""
        self.analyzers.append(analyzer)

    def evaluate(
        self,
        context: TrackContext,
        global_manager=None,
        force_sync: bool = False,
    ) -> AnnotationSpec:
        """
        Run all registered behavior analyzers on `context`, forward active events to
        `global_manager`, and return the combined `AnnotationSpec` for HUD rendering.
        """
        suffixes: list[str] = []
        color = self.default_color

        for analyzer in self.analyzers:
            result = analyzer.analyze(context)
            state_key = (context.global_id, result.name)
            was_active = self._prev_active_state.get(state_key, False)

            if result.is_active:
                if result.label_suffix:
                    suffixes.append(result.label_suffix)
                if result.color_override is not None:
                    color = result.color_override

            if global_manager is not None and result.event_type and (
                result.is_active or was_active or force_sync
            ):
                if hasattr(global_manager, "record_behavior_event"):
                    global_manager.record_behavior_event(
                        global_id=context.global_id,
                        cam_id=context.cam_id,
                        behavior_name=result.name,
                        event_type=result.event_type,
                        is_active=result.is_active,
                        duration_seconds=result.duration_seconds,
                        current_time=context.current_time,
                        details=result.details,
                    )

            self._prev_active_state[state_key] = result.is_active

        label = f"ID:{context.global_id}"
        if suffixes:
            label += " " + " | ".join(suffixes)

        return AnnotationSpec(
            bbox=context.bbox,
            global_id=context.global_id,
            label=label,
            color=color,
        )
