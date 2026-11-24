# behaviors/Calma_NathaliaProducer.py
from behaviors.BehaviorRegistry import BehaviorRegistry
from domain.BehaviorResult import BehaviorResult
from domain.TrackContext import TrackContext
from interfaces.IBehaviorAnalyzer import IBehaviorAnalyzer


@BehaviorRegistry.register("calma_nathalia")
class CalmaNathaliaProducer(IBehaviorAnalyzer):
    """Extracts complete YOLO, DeepSort, Camera Occupancy, and Threshold telemetry for Kafka topic 'Calma_Nathalia'."""

    def __init__(self, config):
        self.config = config
        self.topic = getattr(config, "KAFKA_EVENTS_TOPIC", "Calma_Nathalia")
        self.emit_interval = float(getattr(config, "KAFKA_TELEMETRY_INTERVAL_SEC", 0.25))
        self.first_seen: dict[tuple[int, str], float] = {}
        self.last_emitted: dict[tuple[int, str], float] = {}
        self.last_people_count: dict[str, int] = {}

    def _should_emit(self, key: tuple[int, str], cam_id: str, people_count: int, current_time: float) -> bool:
        prev_t = self.last_emitted.get(key)
        count_changed = self.last_people_count.get(cam_id) != people_count
        if prev_t is None or count_changed or (current_time - prev_t) >= self.emit_interval:
            self.last_emitted[key] = current_time
            self.last_people_count[cam_id] = people_count
            return True
        return False

    def _build_details(self, ctx: TrackContext) -> str:
        conf_pct = ctx.limiares_percentuais.get("yolo_confidence_pct", round(ctx.confidence * 100.0, 2))
        occ_pct = ctx.limiares_percentuais.get("camera_occupancy_pct", 0.0)
        spd = ctx.deepsort_metrics.get("speed_px_per_sec", 0.0)
        return (
            f"Calma_Nathalia | Cam='{ctx.cam_id}' | Pessoas={ctx.people_in_camera} ({occ_pct:.1f}%) | "
            f"ID={ctx.global_id} (Local={ctx.local_id}) | Classe='{ctx.object_class}' | "
            f"Conf={conf_pct:.1f}% | Vel={spd:.1f}px/s"
        )

    def analyze(self, track_context: TrackContext) -> BehaviorResult:
        key = (track_context.global_id, track_context.cam_id)
        self.first_seen.setdefault(key, track_context.current_time)
        elapsed = max(0.0, track_context.current_time - self.first_seen[key])
        emit = self._should_emit(key, track_context.cam_id, track_context.people_in_camera, track_context.current_time)
        payload = track_context.to_telemetry_dict()
        payload["kafka_topic"] = self.topic
        return BehaviorResult(
            name="calma_nathalia",
            is_active=True,
            event_type="CALMA_NATHALIA_TELEMETRY" if emit else None,
            duration_seconds=elapsed,
            details=self._build_details(track_context),
            telemetry=payload,
        )
