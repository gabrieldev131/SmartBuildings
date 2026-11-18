# interfaces/IStateTracker.py
from abc import ABC, abstractmethod
from typing import Tuple


class IStateTracker(ABC):
    """Interface for behavioral state evaluation (moving vs. stopped/loitering)."""

    @abstractmethod
    def update_and_evaluate(self, global_id: int, box: list[float], current_time: float) -> Tuple[bool, float]:
        pass
