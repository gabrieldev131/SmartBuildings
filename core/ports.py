# core/ports.py
"""
Application Ports (Interfaces) following Clean Architecture & Open/Closed Principle (OCP).
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import numpy as np

from core.domain import BehaviorResult, TrackContext


class IFrameCommand(ABC):
    """Command Pattern interface for reading video frames from RTSP, local files, or Kafka."""

    @abstractmethod
    def execute(self) -> Optional[Tuple[str, Any]]:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass


class ICommandFactory(ABC):
    """Abstract Factory Pattern interface for creating picklable IFrameCommand instances."""

    @abstractmethod
    def __call__(self) -> IFrameCommand:
        pass


class IBehaviorAnalyzer(ABC):
    """
    Strategy / Plugin Pattern interface for decoupled behavioral analysis.
    To add a new functionality (e.g., Stopped/Loitering, Restricted Zone, Running, Fall Detection),
    implement `analyze(context: TrackContext) -> BehaviorResult` and register the analyzer in
    `BehaviorPipeline` — no changes to `CameraWorker`, `display.py`, or `GlobalIdentityManager` needed.
    """

    @abstractmethod
    def analyze(self, context: TrackContext) -> BehaviorResult:
        """Evaluate behavior for a tracked person and return a standardized `BehaviorResult`."""
        pass


class IIdentityRepository(ABC):
    """Repository Pattern interface for cross-camera Global Re-ID and CSV export."""

    @abstractmethod
    def get_or_create_global_id(
        self,
        new_feature_vector: np.ndarray,
        bbox: list[float],
        cam_id: str,
        current_time: float,
        active_global_ids: Optional[set[int]] = None,
    ) -> int:
        pass

    @abstractmethod
    def update_existing_identity(
        self,
        global_id: int,
        new_feature_vector: Optional[np.ndarray],
        bbox: list[float],
        cam_id: str,
        current_time: float,
        active_global_ids: Optional[set[int]] = None,
    ) -> Optional[int]:
        pass

    @abstractmethod
    def export_data_to_csv(self, filename: str = "tracking_data_final.csv") -> None:
        pass

    def record_behavior_event(
        self,
        global_id: int,
        cam_id: str,
        behavior_name: str,
        event_type: str,
        is_active: bool,
        duration_seconds: float,
        current_time: float,
        details: str = "",
    ) -> None:
        pass


class IStateTracker(ABC):
    """Legacy interface kept for backward compatibility."""

    @abstractmethod
    def update_and_evaluate(self, global_id: int, box: list[float], current_time: float) -> Tuple[bool, float]:
        pass
