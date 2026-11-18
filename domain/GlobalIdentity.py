# domain/GlobalIdentity.py
from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist


class GlobalIdentity:
    """Domain Entity storing appearance descriptors and per-camera passages for a tracked person."""

    def __init__(self, global_id: int, initial_feature: np.ndarray, bbox: list[float], initial_cam_id: str, start_time: float, max_gallery_size: int = 32):
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
        self.passage_records: list[dict] = [self._make_passage(1, initial_cam_id, "Entrada", start_time, bbox, 0.0)]

    @staticmethod
    def _make_passage(idx: int, cam_id: str, origin: str, ts: float, bbox: list[float], dist: float) -> dict:
        return {
            "passage_index": idx, "camera_id": cam_id, "origin_camera": origin,
            "timestamp_in": ts, "timestamp_out": ts, "entry_bbox": list(bbox),
            "exit_bbox": list(bbox), "stopped_seconds": 0.0, "reid_distance": float(dist),
        }

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-6 else arr

    @property
    def feature_vector(self) -> np.ndarray:
        return self.ema_feature

    def match_score(self, candidate_feature: np.ndarray, cam_id: Optional[str] = None) -> float:
        query = self._normalize(candidate_feature).reshape(1, -1)
        gal_dists = cdist(self._gallery_matrix, query, metric="cosine").ravel()
        min_gal = float(np.min(gal_dists))
        gal_dist = (0.70 * min_gal + 0.30 * float(np.mean(np.sort(gal_dists)[:3]))) if len(gal_dists) >= 3 else min_gal
        ema_refs = [self.ema_feature, self.camera_ema_features[cam_id]] if (cam_id in self.camera_ema_features) else [self.ema_feature, *self.camera_ema_features.values()]
        ema_dist = float(np.min(cdist(np.asarray(ema_refs), query, metric="cosine")))
        return float(np.clip(0.72 * gal_dist + 0.28 * ema_dist, 0.0, 1.0))

    def _update_gallery_matrix(self, unit_feat: np.ndarray, min_dist: float, min_idx: int) -> None:
        if len(self._gallery_matrix) >= 3 and min_dist < 0.08:
            self._gallery_matrix[min_idx] = self._normalize(0.72 * self._gallery_matrix[min_idx] + 0.28 * unit_feat)
        elif len(self._gallery_matrix) >= self.max_gallery_size:
            self._gallery_matrix = np.roll(self._gallery_matrix, -1, axis=0)
            self._gallery_matrix[-1] = unit_feat
        else:
            self._gallery_matrix = np.vstack((self._gallery_matrix, unit_feat.reshape(1, -1)))

    def _integrate_feature(self, unit_feat: np.ndarray, cam_id: str, force_accept: bool = False) -> None:
        if float(np.dot(unit_feat, unit_feat)) <= 0.5:
            return
        query = unit_feat.reshape(1, -1)
        gal_dists = cdist(self._gallery_matrix, query, metric="cosine").ravel()
        min_dist, min_idx = float(np.min(gal_dists)), int(np.argmin(gal_dists))
        ema_dist = float(cdist(self.ema_feature.reshape(1, -1), query, metric="cosine")[0, 0])
        if force_accept or self.update_count <= 3 or ema_dist <= 0.46 or min_dist <= 0.42:
            alpha = 0.60 if self.update_count <= 3 else 0.85
            self.ema_feature = self._normalize(alpha * self.ema_feature + (1.0 - alpha) * unit_feat)
            prev_cam_ema = self.camera_ema_features.get(cam_id)
            self.camera_ema_features[cam_id] = self._normalize(0.80 * prev_cam_ema + 0.20 * unit_feat) if prev_cam_ema is not None else unit_feat.copy()
            self._update_gallery_matrix(unit_feat, min_dist, min_idx)
            self.update_count += 1

    def _record_camera_switch(self, cam_id: str, bbox: list[float], current_time: float, prev_primary: float, dist: float) -> bool:
        old_cam = self.current_camera
        if self.passage_records:
            self.passage_records[-1]["timestamp_out"] = max(self.passage_records[-1]["timestamp_in"], prev_primary)
        self.current_camera = cam_id
        self._last_switch_time = current_time
        self.passage_records.append(self._make_passage(len(self.passage_records) + 1, cam_id, old_cam, current_time, bbox, dist))
        return True

    def _record_same_camera(self, cam_id: str, bbox: list[float], current_time: float, prev_in_cam: Optional[float], dist: float) -> None:
        if prev_in_cam is not None and (current_time - prev_in_cam) > 4.0:
            if self.passage_records:
                self.passage_records[-1]["timestamp_out"] = max(self.passage_records[-1]["timestamp_in"], prev_in_cam)
            self.passage_records.append(self._make_passage(len(self.passage_records) + 1, cam_id, cam_id, current_time, bbox, dist))
        elif self.passage_records:
            self.passage_records[-1]["timestamp_out"] = max(self.passage_records[-1]["timestamp_out"], current_time)
            self.passage_records[-1]["exit_bbox"] = list(bbox)

    def update(self, new_feature_vector: Optional[np.ndarray], bbox: list[float], cam_id: str, current_time: float, switch_cooldown: float = 0.8, reid_distance: float = 0.0, force_transition: bool = False) -> bool:
        if new_feature_vector is not None:
            self._integrate_feature(self._normalize(new_feature_vector), cam_id, force_accept=force_transition)
        prev_in_cam = self.last_seen_per_camera.get(cam_id)
        prev_primary = self.last_seen_per_camera.get(self.current_camera, self.last_seen)
        self.last_bbox = list(bbox)
        self.last_bbox_per_camera[cam_id] = list(bbox)
        self.last_seen = max(self.last_seen, current_time)
        self.last_seen_per_camera[cam_id] = current_time

        if self.current_camera != cam_id:
            if force_transition or (current_time - prev_primary) >= 0.25 or (current_time - self._last_switch_time) >= switch_cooldown:
                return self._record_camera_switch(cam_id, bbox, current_time, prev_primary, reid_distance)
            return False
        self._record_same_camera(cam_id, bbox, current_time, prev_in_cam, reid_distance)
        return False

    def update_stopped_duration(self, cam_id: str, stopped_elapsed: float, current_time: float) -> None:
        for rec in reversed(self.passage_records):
            if rec["camera_id"] == cam_id:
                rec["stopped_seconds"] = max(float(rec["stopped_seconds"]), float(stopped_elapsed))
                rec["timestamp_out"] = max(rec["timestamp_out"], current_time)
                break

    def _format_passage(self, idx: int, rec: dict) -> dict:
        t_in, t_out = float(rec["timestamp_in"]), max(float(rec["timestamp_in"]), float(rec["timestamp_out"]))
        nxt = self.passage_records[idx + 1]["camera_id"] if (idx + 1) < len(self.passage_records) else "Saida"
        return {
            "passage_index": rec["passage_index"], "camera_id": rec["camera_id"],
            "origin_camera": rec.get("origin_camera", "Entrada"), "next_camera": nxt,
            "timestamp_in": t_in, "timestamp_out": t_out, "duration_seconds": max(0.0, t_out - t_in),
            "stopped_seconds": float(rec.get("stopped_seconds", 0.0)),
            "reid_distance": float(rec.get("reid_distance", 0.0)),
            "entry_bbox": rec.get("entry_bbox"), "exit_bbox": rec.get("exit_bbox"),
        }

    def get_raw_history(self) -> list[dict]:
        return [self._format_passage(i, r) for i, r in enumerate(self.passage_records)]
