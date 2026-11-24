# behaviors/BehaviorPipeline.py
from typing import Optional
from behaviors.BehaviorRegistry import BehaviorRegistry
from domain.AnnotationSpec import AnnotationSpec
from domain.BehaviorResult import BehaviorResult
from domain.TrackContext import TrackContext
from interfaces.IBehaviorAnalyzer import IBehaviorAnalyzer


class BehaviorPipeline:
    """Composite runner executing registered `IBehaviorAnalyzer` plugins with zero if/else branches."""

    def __init__(self, config, analyzers: Optional[list[IBehaviorAnalyzer]] = None):
        self.config = config
        self.default_color = getattr(config, "COLOR_MOVING", (0, 255, 0))
        self.analyzers = list(analyzers) if analyzers is not None else BehaviorRegistry.create_all(config)
        self._prev_active_state: dict[tuple[int, str], bool] = {}

    def register(self, analyzer: IBehaviorAnalyzer) -> None:
        self.analyzers.append(analyzer)

    def _notify_manager(self, context: TrackContext, result: BehaviorResult, global_manager, force_sync: bool) -> None:
        state_key = (context.global_id, result.name)
        was_active = self._prev_active_state.get(state_key, False)
        self._prev_active_state[state_key] = result.is_active
        if global_manager is None or not result.event_type:
            return
        if (result.is_active or was_active or force_sync) and hasattr(global_manager, "record_behavior_event"):
            telemetry = result.telemetry or context.to_telemetry_dict()
            global_manager.record_behavior_event(
                global_id=context.global_id, cam_id=context.cam_id, behavior_name=result.name,
                event_type=result.event_type, is_active=result.is_active,
                duration_seconds=result.duration_seconds, current_time=context.current_time,
                details=result.details, telemetry=telemetry,
            )

    def evaluate(self, context: TrackContext, global_manager=None, force_sync: bool = False) -> AnnotationSpec:
        suffixes: list[str] = []
        color = self.default_color
        for analyzer in self.analyzers:
            result = analyzer.analyze(context)
            if result.is_active and result.label_suffix:
                suffixes.append(result.label_suffix)
            if result.is_active and result.color_override is not None:
                color = result.color_override
            self._notify_manager(context, result, global_manager, force_sync)

        label = f"ID:{context.global_id}" + (f" {' | '.join(suffixes)}" if suffixes else "")
        return AnnotationSpec(bbox=context.bbox, global_id=context.global_id, label=label, color=color)
