# core/state_tracker.py
from collections import deque
import numpy as np

class StoppedStateTracker:
    def __init__(self, config):
        self.config = config
        self.position_history = {}
        self.stopped_state = {}
        self.stopped_since = {}

    def update_and_evaluate(self, global_id: int, box: list, current_time: float) -> tuple[bool, float]:
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        if global_id not in self.position_history:
            self.position_history[global_id] = deque(maxlen=90)
            self.stopped_state[global_id] = False
            self.stopped_since[global_id] = current_time

        self.position_history[global_id].append((current_time, center))
        history = self.position_history[global_id]

        if len(history) < 2:
            return False, 0.0

        recent = [pt for pt in history if current_time - pt[0] <= self.config.STOPPED_SECONDS_THRESHOLD]
        if len(recent) < 2:
            return self.stopped_state[global_id], 0.0

        start_pos, end_pos = recent[0][1], recent[-1][1]
        displacement = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)

        currently_stopped = self.stopped_state[global_id]

        if currently_stopped:
            if displacement > self.config.MOVEMENT_BREAKOUT_THRESHOLD:
                self.stopped_state[global_id] = False
                self.stopped_since[global_id] = current_time
        else:
            if displacement < self.config.STOPPED_PIXEL_THRESHOLD:
                if (current_time - self.stopped_since[global_id]) >= self.config.STOPPED_SECONDS_THRESHOLD:
                    self.stopped_state[global_id] = True
            else:
                self.stopped_since[global_id] = current_time

        elapsed = current_time - self.stopped_since[global_id] if self.stopped_state[global_id] else 0.0
        return self.stopped_state[global_id], elapsed