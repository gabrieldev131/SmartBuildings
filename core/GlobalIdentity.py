# core/GlobalIdentity.py
from typing import Optional
import numpy as np


class GlobalIdentity:
    """
    Domain Entity representing a unique person tracked across one or multiple cameras.
    Maintains:
    - Exponential Moving Average (EMA) appearance descriptor + bounded diverse gallery matrix.
    - Detailed per-camera passage history (timestamps, origin/destination cameras, bounding boxes,
      stopped duration, and Re-ID transition distances).
    """

    def __init__(
        self,
        global_id: int,
        initial_feature: np.ndarray,
        bbox: list[float],
        initial_cam_id: str,
        start_time: float,
        max_gallery_size: int = 32,
    ):
        self.global_id = global_id

        unit_feat = self._normalize(initial_feature)
        self.ema_feature: np.ndarray = unit_feat.copy()
        self.camera_ema_features: dict[str, np.ndarray] = {initial_cam_id: unit_feat.copy()}
        self._gallery_matrix: np.ndarray = unit_feat.reshape(1, -1)
        self.max_gallery_size = max_gallery_size
        self.update_count = 1

        self.last_bbox = list(bbox)
        self.last_bbox_per_camera: dict[str, list[float]] = {initial_cam_id: list(bbox)}
        self.first_seen = start_time
        self.last_seen = start_time
        self._last_switch_time = start_time

        self.current_camera = initial_cam_id
        self.last_seen_per_camera: dict[str, float] = {initial_cam_id: start_time}

        # Detailed passage records for every camera visit
        self.passage_records: list[dict] = [
            {
                "passage_index": 1,
                "camera_id": initial_cam_id,
                "origin_camera": "Entrada",
                "timestamp_in": start_time,
                "timestamp_out": start_time,
                "entry_bbox": list(bbox),
                "exit_bbox": list(bbox),
                "stopped_seconds": 0.0,
                "reid_distance": 0.0,
            }
        ]

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-6 else arr

    @property
    def camera_history(self) -> list[list]:
        """Backward-compatible [[cam_id, t_in, t_out], ...] view of passage_records."""
        return [
            [r["camera_id"], r["timestamp_in"], r["timestamp_out"]]
            for r in self.passage_records
        ]

    @property
    def feature_gallery(self) -> list[np.ndarray]:
        """Backward-compatible list view of gallery vectors."""
        return [self._gallery_matrix[i] for i in range(len(self._gallery_matrix))]

    @property
    def feature_vector(self) -> np.ndarray:
        """Normalized EMA representation of this identity."""
        return self.ema_feature

    def match_score(self, candidate_feature: np.ndarray, cam_id: Optional[str] = None) -> float:
        """
        Compute hybrid cosine distance in [0.0, 1.0] combining the diverse multi-pose gallery,
        per-camera EMA appearance memory, and the global EMA centroid.
        """
        candidate_norm = self._normalize(candidate_feature)
        dots = self._gallery_matrix @ candidate_norm
        best_gal_sim = float(np.max(dots))
        ema_sim = float(np.dot(self.ema_feature, candidate_norm))

        if self.camera_ema_features:
            if cam_id is not None and cam_id in self.camera_ema_features:
                cam_sim = float(np.dot(self.camera_ema_features[cam_id], candidate_norm))
                ema_sim = max(ema_sim, cam_sim)
            else:
                best_cam_sim = max(
                    float(np.dot(c_vec, candidate_norm)) for c_vec in self.camera_ema_features.values()
                )
                ema_sim = max(ema_sim, best_cam_sim)

        if len(dots) >= 3:
            k = min(3, len(dots))
            topk_sim = float(np.mean(np.sort(dots)[-k:]))
            gal_sim = 0.70 * best_gal_sim + 0.30 * topk_sim
        else:
            gal_sim = best_gal_sim

        combined_sim = 0.72 * gal_sim + 0.28 * ema_sim
        return float(max(0.0, 1.0 - min(1.0, combined_sim)))

    def _integrate_feature(self, unit_feat: np.ndarray, cam_id: str, force_accept: bool = False) -> None:
        if float(np.dot(unit_feat, unit_feat)) <= 0.5:
            return
        sim_to_ema = float(np.dot(self.ema_feature, unit_feat))
        gal_dots = self._gallery_matrix @ unit_feat
        max_gal_sim = float(np.max(gal_dots))
        max_gal_idx = int(np.argmax(gal_dots))

        # Accept during early frames, forced transitions, or when consistent with EMA/gallery
        if force_accept or self.update_count <= 3 or sim_to_ema >= 0.54 or max_gal_sim >= 0.58:
            alpha = 0.60 if self.update_count <= 3 else 0.85
            self.ema_feature = self._normalize(alpha * self.ema_feature + (1.0 - alpha) * unit_feat)
            if cam_id in self.camera_ema_features:
                self.camera_ema_features[cam_id] = self._normalize(
                    0.80 * self.camera_ema_features[cam_id] + 0.20 * unit_feat
                )
            else:
                self.camera_ema_features[cam_id] = unit_feat.copy()

            # Diversity-preserving gallery: blend in-place if near-duplicate so standing still never erases other cameras/poses
            if len(self._gallery_matrix) >= 3 and max_gal_sim > 0.92:
                self._gallery_matrix[max_gal_idx] = self._normalize(
                    0.72 * self._gallery_matrix[max_gal_idx] + 0.28 * unit_feat
                )
            elif len(self._gallery_matrix) >= self.max_gallery_size:
                self._gallery_matrix = np.roll(self._gallery_matrix, -1, axis=0)
                self._gallery_matrix[-1] = unit_feat
            else:
                self._gallery_matrix = np.vstack((self._gallery_matrix, unit_feat.reshape(1, -1)))
            self.update_count += 1

    def update(
        self,
        new_feature_vector: Optional[np.ndarray],
        bbox: list[float],
        cam_id: str,
        current_time: float,
        switch_cooldown: float = 1.0,
        reid_distance: float = 0.0,
        force_transition: bool = False,
    ) -> bool:
        if new_feature_vector is not None:
            unit_feat = self._normalize(new_feature_vector)
            self._integrate_feature(unit_feat, cam_id, force_accept=force_transition)

        prev_seen_in_cam = self.last_seen_per_camera.get(cam_id)
        prev_seen_primary = self.last_seen_per_camera.get(self.current_camera, self.last_seen)

        self.last_bbox = list(bbox)
        self.last_bbox_per_camera[cam_id] = list(bbox)
        self.last_seen = max(self.last_seen, current_time)
        self.last_seen_per_camera[cam_id] = current_time

        if self.current_camera != cam_id:
            time_since_primary_seen = current_time - prev_seen_primary
            time_since_last_switch = current_time - self._last_switch_time

            # Allow immediate transition when matched via get_or_create_global_id (force_transition=True)
            # or when the person left the previous camera, while preventing rapid ping-pong on overlap
            can_switch = (
                force_transition
                or (time_since_primary_seen >= 0.25)
                or (time_since_last_switch >= switch_cooldown)
            )
            if can_switch:
                old_cam = self.current_camera
                if self.passage_records:
                    self.passage_records[-1]["timestamp_out"] = max(
                        self.passage_records[-1]["timestamp_in"], prev_seen_primary
                    )
                self.current_camera = cam_id
                self._last_switch_time = current_time
                self.passage_records.append(
                    {
                        "passage_index": len(self.passage_records) + 1,
                        "camera_id": cam_id,
                        "origin_camera": old_cam,
                        "timestamp_in": current_time,
                        "timestamp_out": current_time,
                        "entry_bbox": list(bbox),
                        "exit_bbox": list(bbox),
                        "stopped_seconds": 0.0,
                        "reid_distance": float(reid_distance),
                    }
                )
                return True
        else:
            # Same camera: if person had left this camera for > 4.0s and re-entered, record new passage
            if prev_seen_in_cam is not None and (current_time - prev_seen_in_cam) > 4.0:
                if self.passage_records:
                    self.passage_records[-1]["timestamp_out"] = max(
                        self.passage_records[-1]["timestamp_in"], prev_seen_in_cam
                    )
                self.passage_records.append(
                    {
                        "passage_index": len(self.passage_records) + 1,
                        "camera_id": cam_id,
                        "origin_camera": cam_id,
                        "timestamp_in": current_time,
                        "timestamp_out": current_time,
                        "entry_bbox": list(bbox),
                        "exit_bbox": list(bbox),
                        "stopped_seconds": 0.0,
                        "reid_distance": float(reid_distance),
                    }
                )
            elif self.passage_records:
                self.passage_records[-1]["timestamp_out"] = max(
                    self.passage_records[-1]["timestamp_out"], current_time
                )
                self.passage_records[-1]["exit_bbox"] = list(bbox)

        return False

    def update_stopped_duration(self, cam_id: str, stopped_elapsed: float, current_time: float) -> None:
        """Record maximum stopped/loitering duration during the active passage in `cam_id`."""
        for rec in reversed(self.passage_records):
            if rec["camera_id"] == cam_id:
                rec["stopped_seconds"] = max(float(rec.get("stopped_seconds", 0.0)), float(stopped_elapsed))
                rec["timestamp_out"] = max(rec["timestamp_out"], current_time)
                break

    def get_raw_history(self) -> list[dict]:
        """Return copy of all per-camera passage records with accurate timestamps."""
        raw_data = []
        for idx, rec in enumerate(self.passage_records):
            t_in = float(rec["timestamp_in"])
            t_out = max(t_in, float(rec["timestamp_out"]))
            next_cam = (
                self.passage_records[idx + 1]["camera_id"]
                if (idx + 1) < len(self.passage_records)
                else "Saida"
            )
            raw_data.append(
                {
                    "passage_index": rec["passage_index"],
                    "camera_id": rec["camera_id"],
                    "origin_camera": rec.get("origin_camera", "Entrada"),
                    "next_camera": next_cam,
                    "timestamp_in": t_in,
                    "timestamp_out": t_out,
                    "duration_seconds": max(0.0, t_out - t_in),
                    "stopped_seconds": float(rec.get("stopped_seconds", 0.0)),
                    "reid_distance": float(rec.get("reid_distance", 0.0)),
                    "entry_bbox": rec.get("entry_bbox"),
                    "exit_bbox": rec.get("exit_bbox"),
                }
            )
        return raw_data
