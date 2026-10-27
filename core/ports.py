# core/ports.py
"""
Application Ports (Interfaces) following Clean Architecture & Dependency Inversion Principle (DIP).
Outer layers (extraction, vision, ui) implement these abstractions while core business logic
depends only on these contracts.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import numpy as np


class IFrameCommand(ABC):
    """
    Command Pattern interface for frame extraction from heterogeneous video sources
    (RTSP cameras, local video files, Apache Kafka streams).
    """

    @abstractmethod
    def execute(self) -> Optional[Tuple[str, Any]]:
        """Execute a single frame extraction step and return (cam_id, frame) or None."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release underlying I/O handles, network sockets, or video capture descriptors."""
        pass


class ICommandFactory(ABC):
    """
    Abstract Factory Pattern interface for creating picklable IFrameCommand instances
    across process boundaries.
    """

    @abstractmethod
    def __call__(self) -> IFrameCommand:
        """Instantiate and return a concrete IFrameCommand."""
        pass


class IMultiObjectTracker(ABC):
    """
    Strategy / Adapter Pattern interface for multi-target within-camera tracking.
    """

    @abstractmethod
    def update(self, boxes_xyxy: np.ndarray, confs: np.ndarray, embeds: np.ndarray) -> list[Any]:
        """Update tracker state with current frame detections and appearance embeddings."""
        pass


class IIdentityRepository(ABC):
    """
    Repository Pattern interface for cross-camera Global Re-Identification and trajectory persistence.
    """

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
    ) -> None:
        pass

    @abstractmethod
    def export_data_to_csv(self, filename: str = "tracking_data_final.csv") -> None:
        pass


class IStateTracker(ABC):
    """Interface for behavioral state evaluation (moving vs. stopped/loitering)."""

    @abstractmethod
    def update_and_evaluate(self, global_id: int, box: list[float], current_time: float) -> Tuple[bool, float]:
        pass
