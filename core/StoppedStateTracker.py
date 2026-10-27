# core/StoppedStateTracker.py
from collections import deque
from core.ports import IStateTracker


class StoppedStateTracker(IStateTracker):
    """
    Behavioral Finite State Machine evaluating whether each tracked person is
    moving or stopped/loitering using spatial hysteresis and sliding time windows.
    """

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

    def _cleanup_stale(self, current_time: float, max_idle: float = 60.0) -> None:
        if (current_time - self._last_cleanup) < 10.0:
            return
        self._last_cleanup = current_time
        stale_ids = [gid for gid, ts in self.last_seen.items() if (current_time - ts) > max_idle]
        for gid in stale_ids:
            self.position_history.pop(gid, None)
            self.stopped_state.pop(gid, None)
            self.stopped_since.pop(gid, None)
            self.last_seen.pop(gid, None)

    def update_and_evaluate(self, global_id: int, box: list[float], current_time: float) -> tuple[bool, float]:
        self._cleanup_stale(current_time)
        x1, y1, x2, y2 = box
        center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        if global_id not in self.position_history:
            self.position_history[global_id] = deque(maxlen=120)
            self.stopped_state[global_id] = False
            self.stopped_since[global_id] = current_time

        self.last_seen[global_id] = current_time
        history = self.position_history[global_id]
        history.append((current_time, center))

        while len(history) > 0 and (current_time - history[0][0]) > self._stopped_sec:
            history.popleft()

        if len(history) < 2:
            return self.stopped_state[global_id], 0.0

        start_pos = history[0][1]
        end_pos = history[-1][1]

        squared_displacement = (end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2
        currently_stopped = self.stopped_state[global_id]

        if currently_stopped:
            if squared_displacement > self._breakout_px_sq:
                self.stopped_state[global_id] = False
                self.stopped_since[global_id] = current_time
        else:
            if squared_displacement < self._stopped_px_sq:
                if (current_time - self.stopped_since[global_id]) >= self._stopped_sec:
                    self.stopped_state[global_id] = True
            else:
                self.stopped_since[global_id] = current_time

        elapsed = (current_time - self.stopped_since[global_id]) if self.stopped_state[global_id] else 0.0
        return self.stopped_state[global_id], elapsed
