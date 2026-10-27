# core/GlobalIdentityManager.py
import csv
import logging
import threading
from datetime import datetime
from typing import Optional
import numpy as np

from core.GlobalIdentity import GlobalIdentity
from core.CameraClusterManager import CameraClusterManager
from core.ports import IIdentityRepository


class GlobalIdentityManager(IIdentityRepository):
    """
    Thread-safe Repository and Domain Service managing cross-camera person Re-Identification,
    spatial-temporal transition validation, and automatic Union-Find camera clustering.
    """

    def __init__(self, config=None):
        self.config = config

        self.intra_camera_threshold = getattr(config, "SIMILARITY_THRESHOLD", 0.36)
        self.inter_camera_threshold = getattr(config, "INTER_CAMERA_THRESHOLD", 0.44)
        self.max_time_lost = getattr(config, "MAX_TIME_LOST", 12.0)

        max_dist = getattr(config, "MAX_SPATIAL_DISTANCE", 150)
        self.max_spatial_distance_sq = float(max_dist ** 2)

        self.switch_cooldown = getattr(config, "CAMERA_SWITCH_COOLDOWN", 2.0)
        self.cluster_manager = CameraClusterManager()
        self.auto_cluster_threshold = getattr(config, "AUTO_CLUSTER_TIME_THRESHOLD", 10.0)

        self.frame_w = getattr(config, "PROCESSING_WIDTH", 640)
        self.frame_h = getattr(config, "PROCESSING_HEIGHT", 480)

        self.identities: dict[int, GlobalIdentity] = {}
        self.next_global_id = 1
        self._lock = threading.RLock()
        self._last_cleanup_time = 0.0

    @staticmethod
    def _get_center(bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def _is_near_edge(self, bbox: list[float]) -> bool:
        x1, y1, x2, y2 = bbox
        margin_x = self.frame_w * 0.04
        margin_y = self.frame_h * 0.04
        return (
            (x1 < margin_x)
            or (y1 < margin_y)
            or (x2 > self.frame_w - margin_x)
            or (y2 > self.frame_h - margin_y)
        )

    def _cleanup_old_identities(self, current_time: float) -> None:
        # Throttle cleanup scan to once every 2 seconds to avoid per-frame dict iteration overhead
        if (current_time - self._last_cleanup_time) < 2.0:
            return
        self._last_cleanup_time = current_time
        expiry = self.max_time_lost * 3.0
        ids_to_remove = [
            gid for gid, ident in self.identities.items()
            if (current_time - ident.last_seen) > expiry
        ]
        for gid in ids_to_remove:
            del self.identities[gid]

    def _check_and_update_clusters(self, identity_obj: GlobalIdentity, cam_id: str, current_time: float) -> None:
        for old_cam, last_t in identity_obj.last_seen_per_camera.items():
            if old_cam != cam_id and (current_time - last_t) <= self.auto_cluster_threshold:
                if self.cluster_manager.union(cam_id, old_cam):
                    root_cluster = self.cluster_manager.find(cam_id)
                    logging.info(f"Mapeamento: Câmaras '{cam_id}' e '{old_cam}' fundidas no 'Ambiente_{root_cluster}'.")

    def update_existing_identity(
        self,
        global_id: int,
        new_feature_vector: Optional[np.ndarray],
        bbox: list[float],
        cam_id: str,
        current_time: float,
    ) -> None:
        with self._lock:
            identity = self.identities.get(global_id)
            if identity is not None:
                self._check_and_update_clusters(identity, cam_id, current_time)
                feat_to_update = None if (new_feature_vector is None or self._is_near_edge(bbox)) else new_feature_vector
                identity.update(feat_to_update, bbox, cam_id, current_time, switch_cooldown=self.switch_cooldown)

    def get_or_create_global_id(
        self,
        new_feature_vector: np.ndarray,
        bbox: list[float],
        cam_id: str,
        current_time: float,
        active_global_ids: Optional[set[int]] = None,
    ) -> int:
        if active_global_ids is None:
            active_global_ids = set()

        best_match_id = None
        best_distance = float("inf")
        cx, cy = self._get_center(bbox)

        with self._lock:
            self._cleanup_old_identities(current_time)

            for global_id, identity_obj in self.identities.items():
                # Never assign a Global ID that is already actively tracked in the same camera frame
                if global_id in active_global_ids:
                    continue

                time_lost = current_time - identity_obj.last_seen
                if time_lost > self.max_time_lost:
                    continue

                appearance_dist = identity_obj.match_score(new_feature_vector)
                is_same_camera = (identity_obj.current_camera == cam_id)

                if is_same_camera:
                    # Combine appearance similarity with spatial-temporal proximity bonus for recently lost tracks
                    old_cx, old_cy = self._get_center(identity_obj.last_bbox)
                    dist_sq = (cx - old_cx) ** 2 + (cy - old_cy) ** 2
                    if time_lost < 1.5 and dist_sq <= self.max_spatial_distance_sq:
                        # Small proximity bonus when re-acquiring a person right where they just were
                        effective_dist = max(0.0, appearance_dist - 0.04)
                    else:
                        effective_dist = appearance_dist

                    if effective_dist < best_distance and effective_dist < self.intra_camera_threshold:
                        best_distance = effective_dist
                        best_match_id = global_id
                else:
                    if appearance_dist < best_distance and appearance_dist < self.inter_camera_threshold:
                        best_distance = appearance_dist
                        best_match_id = global_id

            is_edge = self._is_near_edge(bbox)
            if best_match_id is not None:
                identity = self.identities[best_match_id]
                self._check_and_update_clusters(identity, cam_id, current_time)

                feat_to_update = None if is_edge else new_feature_vector
                mudou_de_camera = identity.update(feat_to_update, bbox, cam_id, current_time, self.switch_cooldown)
                if mudou_de_camera:
                    logging.info(
                        f"Re-ID Evento: ID {best_match_id} moveu-se para a câmara '{cam_id}' (Distância: {best_distance:.3f})"
                    )
                return best_match_id

            new_id = self.next_global_id
            nova_identidade = GlobalIdentity(
                global_id=new_id,
                initial_feature=new_feature_vector,
                bbox=bbox,
                initial_cam_id=cam_id,
                start_time=current_time,
            )

            self.identities[new_id] = nova_identidade
            self.next_global_id += 1
            logging.info(f"Re-ID Evento: Nova pessoa detectada -> Criado ID {new_id} na câmara '{cam_id}'")
            return new_id

    def _aggregate_history_by_clusters(self, global_id: int, raw_history: list) -> list[dict]:
        if not raw_history:
            return []

        aggregated = []
        current_cluster = self.cluster_manager.find(raw_history[0]["camera_id"])
        t_in = raw_history[0]["timestamp_in"]
        t_out = raw_history[0]["timestamp_out"]

        for record in raw_history[1:]:
            cluster = self.cluster_manager.find(record["camera_id"])
            r_in = record["timestamp_in"]
            r_out = record["timestamp_out"]

            if cluster == current_cluster:
                t_out = max(t_out, r_out)
            else:
                aggregated.append({
                    "global_id": global_id,
                    "camera_id": current_cluster,
                    "ambiente_id": f"Ambiente_{current_cluster}",
                    "timestamp_in": t_in,
                    "timestamp_out": t_out,
                    "duration_seconds": t_out - t_in,
                })
                current_cluster = cluster
                t_in = r_in
                t_out = r_out

        aggregated.append({
            "global_id": global_id,
            "camera_id": current_cluster,
            "ambiente_id": f"Ambiente_{current_cluster}",
            "timestamp_in": t_in,
            "timestamp_out": t_out,
            "duration_seconds": t_out - t_in,
        })
        return aggregated

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object.strftime("%H:%M:%S:%d/%m/%Y")

    def get_identity_history(self, global_id: int) -> list[dict]:
        with self._lock:
            identity = self.identities.get(global_id)
            if not identity:
                return []
            raw = identity.get_raw_history()

        clustered = self._aggregate_history_by_clusters(global_id, raw)

        formatted = []
        for record in clustered:
            formatted.append({
                "Ambiente": record["ambiente_id"],
                "Entrada": self._format_timestamp(record["timestamp_in"]),
                "Saida": self._format_timestamp(record["timestamp_out"]),
            })
        return formatted

    def export_data_to_csv(self, filename: str = "tracking_data.csv") -> None:
        with self._lock:
            if not self.identities:
                return

            all_records = []
            for gid, identity in self.identities.items():
                raw = identity.get_raw_history()
                clustered = self._aggregate_history_by_clusters(gid, raw)
                all_records.extend(clustered)

        if not all_records:
            return
        keys = list(all_records[0].keys())

        with open(filename, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_records)
