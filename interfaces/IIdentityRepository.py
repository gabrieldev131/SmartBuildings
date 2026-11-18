# interfaces/IIdentityRepository.py
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IIdentityRepository(ABC):
    """Repository Pattern interface for cross-camera Global Re-ID and data export (DIP & ISP)."""

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
    def export_data(self) -> None:
        pass
