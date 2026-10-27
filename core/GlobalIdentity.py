# core/GlobalIdentity.py
import time
from typing import Optional
import numpy as np


class GlobalIdentity:
    """
    Domain Entity representing a unique person across multiple cameras.
    Maintains both an Exponential Moving Average (EMA) descriptor and a bounded,
    quality-gated gallery matrix for fast vectorized cosine similarity matching.
    """

    def __init__(
        self,
        global_id: int,
        initial_feature: np.ndarray,
        bbox: list[float],
        initial_cam_id: str,
        start_time: float,
        max_gallery_size: int = 10,
    ):
        self.global_id = global_id

        unit_feat = self._normalize(initial_feature)
        self.ema_feature: np.ndarray = unit_feat.copy()
        self._gallery_matrix: np.ndarray = unit_feat.reshape(1, -1)
        self.max_gallery_size = max_gallery_size

        self.last_bbox = bbox
        self.first_seen = start_time
        self.last_seen = start_time

        self.current_camera = initial_cam_id
        self.last_seen_per_camera = {initial_cam_id: start_time}
        self.camera_history = [[initial_cam_id, start_time, None]]

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-6 else arr

    @property
    def feature_gallery(self) -> list[np.ndarray]:
        """Backward-compatible list view of gallery vectors."""
        return [self._gallery_matrix[i] for i in range(len(self._gallery_matrix))]

    @property
    def feature_vector(self) -> np.ndarray:
        """Normalized EMA representation of this identity."""
        return self.ema_feature

    def match_score(self, candidate_feature: np.ndarray) -> float:
        """
        Compute hybrid cosine distance in [0.0, 1.0] combining the best gallery sample
        (to handle pose/angle variations) and the EMA centroid (to suppress single-frame outliers).
        """
        candidate_norm = self._normalize(candidate_feature)
        dots = self._gallery_matrix @ candidate_norm
        best_gal_sim = float(np.max(dots))
        ema_sim = float(np.dot(self.ema_feature, candidate_norm))

        # Weighted similarity: 70% best view in gallery + 30% stable EMA centroid
        combined_sim = 0.70 * best_gal_sim + 0.30 * ema_sim
        return float(max(0.0, 1.0 - min(1.0, combined_sim)))

    def update(
        self,
        new_feature_vector: Optional[np.ndarray],
        bbox: list[float],
        cam_id: str,
        current_time: float,
        switch_cooldown: float = 2.0,
    ) -> bool:
        if new_feature_vector is not None:
            unit_feat = self._normalize(new_feature_vector)
            if float(np.dot(unit_feat, unit_feat)) > 0.5:
                # Reject severe outlier crops (e.g., sudden occlusion by another person/object)
                sim_to_ema = float(np.dot(self.ema_feature, unit_feat))
                if sim_to_ema >= 0.35 or len(self._gallery_matrix) < 3:
                    self.ema_feature = self._normalize(0.85 * self.ema_feature + 0.15 * unit_feat)
                    if len(self._gallery_matrix) >= self.max_gallery_size:
                        # Replace oldest entry while preserving diverse views
                        self._gallery_matrix = np.roll(self._gallery_matrix, -1, axis=0)
                        self._gallery_matrix[-1] = unit_feat
                    else:
                        self._gallery_matrix = np.vstack((self._gallery_matrix, unit_feat.reshape(1, -1)))

        self.last_bbox = bbox
        self.last_seen = current_time
        self.last_seen_per_camera[cam_id] = current_time

        if self.current_camera != cam_id:
            time_since_primary = current_time - self.last_seen_per_camera.get(self.current_camera, 0.0)
            if time_since_primary > switch_cooldown:
                time_of_exit = self.last_seen_per_camera[self.current_camera]
                if self.camera_history:
                    self.camera_history[-1][2] = time_of_exit
                self.current_camera = cam_id
                self.camera_history.append([cam_id, time_of_exit, None])
                return True
        return False

    def get_raw_history(self) -> list[dict]:
        raw_data = []
        now = time.time()
        for record in self.camera_history:
            cam_id, t_in, t_out = record
            raw_data.append({
                "camera_id": cam_id,
                "timestamp_in": t_in,
                "timestamp_out": now if t_out is None else t_out,
            })
        return raw_data
