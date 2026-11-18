# behaviors/StoppedStateTracker.py
from collections import deque
from behaviors.BehaviorRegistry import BehaviorRegistry
from domain.BehaviorResult import BehaviorResult
from domain.TrackContext import TrackContext
from interfaces.IBehaviorAnalyzer import IBehaviorAnalyzer
from interfaces.IStateTracker import IStateTracker


@BehaviorRegistry.register("stopped")
class StoppedStateTracker(IBehaviorAnalyzer, IStateTracker):
    """Finite State Machine evaluating moving vs. stopped/loitering state (SRP & OCP)."""

    def __init__(self, config):
        self.config = config
        self.position_history: dict[int, deque] = {}
        self.stopped_state: dict[int, bool] = {}
        self.stopped_since: dict[int, float] = {}
        self.last_seen: dict[int, float] = {}
        self._last_cleanup = 0.0
        self._stopped_sec = getattr(config, "STOPPED_SECONDS_THRESHOLD", 3)
        self._stopped_px_sq = getattr(config, "STOPPED_PIXEL_THRESHOLD", 15) ** 2
        self._breakout_px_sq = getattr(config, "MOVEMENT_BREAKOUT_THRESHOLD", 30) ** 2
        self._color_stopped = getattr(config, "COLOR_STOPPED", (0, 0, 255))

    def _cleanup_stale(self, current_time: float, max_idle: float = 60.0) -> None:
        if (current_time - self._last_cleanup) < 10.0:
            return
        self._last_cleanup = current_time
        for gid in [g for g, ts in self.last_seen.items() if (current_time - ts) > max_idle]:
            self.position_history.pop(gid, None)
            self.stopped_state.pop(gid, None)
            self.stopped_since.pop(gid, None)
            self.last_seen.pop(gid, None)

    def _record_position(self, global_id: int, box: list[float], current_time: float) -> deque:
        if global_id not in self.position_history:
            self.position_history[global_id] = deque(maxlen=120)
            self.stopped_state[global_id] = False
            self.stopped_since[global_id] = current_time
        self.last_seen[global_id] = current_time
        history = self.position_history[global_id]
        history.append((current_time, ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)))
        while history and (current_time - history[0][0]) > self._stopped_sec:
            history.popleft()
        return history

    def _update_hysteresis(self, global_id: int, sq_disp: float, current_time: float) -> None:
        if self.stopped_state[global_id]:
            if sq_disp > self._breakout_px_sq:
                self.stopped_state[global_id] = False
                self.stopped_since[global_id] = current_time
        elif sq_disp < self._stopped_px_sq:
            if (current_time - self.stopped_since[global_id]) >= self._stopped_sec:
                self.stopped_state[global_id] = True
        else:
            self.stopped_since[global_id] = current_time

    def update_and_evaluate(self, global_id: int, box: list[float], current_time: float) -> tuple[bool, float]:
        self._cleanup_stale(current_time)
        history = self._record_position(global_id, box, current_time)
        if len(history) < 2:
            return self.stopped_state[global_id], 0.0
        dx = history[-1][1][0] - history[0][1][0]
        dy = history[-1][1][1] - history[0][1][1]
        self._update_hysteresis(global_id, dx * dx + dy * dy, current_time)
        is_stopped = self.stopped_state[global_id]
        elapsed = (current_time - self.stopped_since[global_id]) if is_stopped else 0.0
        return is_stopped, elapsed

    def analyze(self, context: TrackContext) -> BehaviorResult:
        is_stopped, elapsed = self.update_and_evaluate(context.global_id, context.bbox, context.current_time)
        details = f"Pessoa ID {context.global_id} ficou parada em '{context.cam_id}' por {elapsed:.1f}s" if is_stopped else ""
        return BehaviorResult(
            name="stopped",
            is_active=is_stopped,
            label_suffix=f"PARADO {int(elapsed)}s" if is_stopped else "",
            color_override=self._color_stopped if is_stopped else None,
            event_type="PERSON_STOPPED",
            duration_seconds=elapsed,
            details=details,
        )
