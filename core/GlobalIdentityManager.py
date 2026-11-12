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
    spatial-temporal transition validation, automatic Union-Find camera clustering,
    and comprehensive chronological event + trajectory CSV persistence.
    """

    CSV_FIELDS = [
        "event_index",
        "event_type",
        "global_id",
        "camera_id",
        "origin_camera",
        "next_camera",
        "ambiente_id",
        "merged_cameras",
        "cameras_path",
        "timestamp_in",
        "timestamp_out",
        "datetime_in",
        "datetime_out",
        "duration_seconds",
        "stopped_seconds",
        "reid_distance",
        "details",
    ]

    def __init__(self, config=None):
        self.config = config

        self.intra_camera_threshold = getattr(config, "SIMILARITY_THRESHOLD", 0.32)
        self.inter_camera_threshold = getattr(config, "INTER_CAMERA_THRESHOLD", 0.36)
        self.simultaneous_inter_camera_threshold = getattr(
            config, "SIMULTANEOUS_INTER_CAMERA_THRESHOLD", 0.30
        )
        self.max_time_lost = getattr(config, "MAX_TIME_LOST", 120.0)

        max_dist = getattr(config, "MAX_SPATIAL_DISTANCE", 160)
        self.max_spatial_distance = float(max_dist)
        self.max_spatial_distance_sq = float(max_dist ** 2)

        self.switch_cooldown = getattr(config, "CAMERA_SWITCH_COOLDOWN", 0.8)
        self.cluster_manager = CameraClusterManager()
        self.auto_cluster_threshold = getattr(config, "AUTO_CLUSTER_TIME_THRESHOLD", 15.0)

        self.frame_w = getattr(config, "PROCESSING_WIDTH", 640)
        self.frame_h = getattr(config, "PROCESSING_HEIGHT", 480)

        self.identities: dict[int, GlobalIdentity] = {}
        self.archived_identities: dict[int, GlobalIdentity] = {}
        self.event_log: list[dict] = []
        self._active_stopped_events: dict[tuple[int, str], dict] = {}

        self.next_global_id = 1
        self._lock = threading.RLock()
        self._last_cleanup_time = 0.0

    @staticmethod
    def _get_center(bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def _is_near_edge(self, bbox: list[float]) -> bool:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        margin_x = self.frame_w * 0.03
        margin_y = self.frame_h * 0.03
        clipped_x = ((x1 < margin_x) or (x2 > self.frame_w - margin_x)) and (w < self.frame_w * 0.25)
        clipped_y = ((y1 < margin_y) or (y2 > self.frame_h - margin_y)) and (h < self.frame_h * 0.20)
        return clipped_x or clipped_y

    def _cleanup_old_identities(self, current_time: float) -> None:
        if (current_time - self._last_cleanup_time) < 2.0:
            return
        self._last_cleanup_time = current_time
        expiry = self.max_time_lost * 3.0
        ids_to_archive = [
            gid for gid, ident in self.identities.items()
            if (current_time - ident.last_seen) > expiry
        ]
        for gid in ids_to_archive:
            # Archive rather than delete so full history is preserved for CSV export
            self.archived_identities[gid] = self.identities.pop(gid)

    def _check_and_update_clusters(
        self,
        identity_obj: GlobalIdentity,
        cam_id: str,
        current_time: float,
        reid_distance: float = 0.0,
    ) -> None:
        self.cluster_manager.find(cam_id)
        for old_cam, last_t in list(identity_obj.last_seen_per_camera.items()):
            gap = current_time - last_t
            if old_cam != cam_id and 0.0 <= gap <= self.auto_cluster_threshold:
                if self.cluster_manager.union(cam_id, old_cam):
                    root_cluster = self.cluster_manager.find(cam_id)
                    merged_label = self.cluster_manager.get_cluster_label(cam_id)
                    msg = (
                        f"Câmaras '{old_cam}' e '{cam_id}' fundidas no 'Ambiente_{root_cluster}' "
                        f"(grupo=[{merged_label}]) via transição da Pessoa ID {identity_obj.global_id} "
                        f"em {gap:.2f}s (dist={reid_distance:.3f})"
                    )
                    logging.info(f"Mapeamento: {msg}")
                    self.event_log.append(
                        {
                            "event_type": "CAMERA_MERGE",
                            "global_id": identity_obj.global_id,
                            "camera_id": cam_id,
                            "origin_camera": old_cam,
                            "next_camera": cam_id,
                            "ambiente_id": f"Ambiente_{root_cluster}",
                            "merged_cameras": merged_label,
                            "cameras_path": f"{old_cam} -> {cam_id}",
                            "timestamp_in": last_t,
                            "timestamp_out": current_time,
                            "duration_seconds": max(0.0, gap),
                            "stopped_seconds": 0.0,
                            "reid_distance": float(reid_distance),
                            "details": msg,
                        }
                    )

    def _find_best_matching_identity(
        self,
        new_feature_vector: np.ndarray,
        bbox: list[float],
        cam_id: str,
        current_time: float,
        active_global_ids: set[int],
        max_candidate_id: Optional[int] = None,
    ) -> tuple[Optional[int], float, float]:
        best_match_id = None
        best_distance = float("inf")
        best_raw_app_dist = 0.0
        cx, cy = self._get_center(bbox)
        new_h = max(16.0, float(bbox[3] - bbox[1]))
        target_cluster = self.cluster_manager.find(cam_id)

        for global_id, identity_obj in self.identities.items():
            if max_candidate_id is not None and global_id >= max_candidate_id:
                continue
            if global_id in active_global_ids:
                continue

            time_lost = current_time - identity_obj.last_seen
            if time_lost > self.max_time_lost:
                continue

            appearance_dist = identity_obj.match_score(new_feature_vector, cam_id=cam_id)
            is_same_camera = (identity_obj.current_camera == cam_id)

            if is_same_camera:
                ref_bbox = identity_obj.last_bbox_per_camera.get(cam_id, identity_obj.last_bbox)
                old_cx, old_cy = self._get_center(ref_bbox)
                old_h = max(16.0, float(ref_bbox[3] - ref_bbox[1]))
                dist = float(np.hypot(cx - old_cx, cy - old_cy))
                scale_ratio = min(new_h, old_h) / max(new_h, old_h)

                # Physical speed & scale gate for recently lost tracks in the same camera
                if time_lost < 3.0:
                    max_allowed_dist = max(self.max_spatial_distance, 220.0 * max(0.40, time_lost))
                    if dist > max_allowed_dist or scale_ratio < 0.45:
                        continue

                if time_lost < 2.0 and dist <= self.max_spatial_distance and scale_ratio >= 0.60:
                    effective_dist = max(0.0, appearance_dist - 0.035)
                else:
                    effective_dist = appearance_dist

                if effective_dist < best_distance and effective_dist < self.intra_camera_threshold:
                    best_distance = effective_dist
                    best_raw_app_dist = appearance_dist
                    best_match_id = global_id
            else:
                # Cross-camera matching
                seen_in_primary = identity_obj.last_seen_per_camera.get(
                    identity_obj.current_camera, identity_obj.last_seen
                )
                time_since_primary = current_time - seen_in_primary
                exiting_primary = self._is_near_edge(identity_obj.last_bbox)

                if time_since_primary < 0.35 and not exiting_primary:
                    thresh = self.simultaneous_inter_camera_threshold
                else:
                    thresh = self.inter_camera_threshold

                # Temporal handoff bonus when person recently left another camera or moves within same cluster
                same_cluster = (self.cluster_manager.find(identity_obj.current_camera) == target_cluster)
                if 0.10 <= time_since_primary <= self.auto_cluster_threshold or (
                    same_cluster and time_since_primary <= 25.0
                ):
                    effective_dist = max(0.0, appearance_dist - 0.030)
                else:
                    effective_dist = appearance_dist

                if effective_dist < best_distance and effective_dist < thresh:
                    best_distance = effective_dist
                    best_raw_app_dist = appearance_dist
                    best_match_id = global_id

        return best_match_id, best_distance, best_raw_app_dist

    def _merge_provisional_into_existing(
        self,
        prov_id: int,
        target_id: int,
        new_feature_vector: Optional[np.ndarray],
        bbox: list[float],
        cam_id: str,
        current_time: float,
        reid_distance: float,
    ) -> int:
        prov_ident = self.identities.get(prov_id)
        target_ident = self.identities.get(target_id)
        if prov_ident is None or target_ident is None:
            return target_id

        old_cam = target_ident.current_camera
        prev_seen = target_ident.last_seen_per_camera.get(old_cam, target_ident.last_seen)
        entry_ts = prov_ident.first_seen

        if old_cam != cam_id:
            self._check_and_update_clusters(target_ident, cam_id, entry_ts, reid_distance=reid_distance)

        mudou_de_camera = target_ident.update(
            new_feature_vector,
            bbox,
            cam_id,
            current_time,
            switch_cooldown=self.switch_cooldown,
            reid_distance=reid_distance,
            force_transition=(old_cam != cam_id),
        )
        # Preserve accurate entry timestamp & entry bbox from when the provisional track first appeared
        if target_ident.passage_records and target_ident.passage_records[-1]["camera_id"] == cam_id:
            target_ident.passage_records[-1]["timestamp_in"] = min(
                target_ident.passage_records[-1]["timestamp_in"], entry_ts
            )
            if prov_ident.passage_records:
                target_ident.passage_records[-1]["entry_bbox"] = list(
                    prov_ident.passage_records[0]["entry_bbox"]
                )

        # Remove provisional PERSON_DETECTED event for prov_id and rebind any stopped events
        self.event_log = [
            ev
            for ev in self.event_log
            if not (ev.get("global_id") == prov_id and ev.get("event_type") == "PERSON_DETECTED")
        ]
        for ev in self.event_log:
            if ev.get("global_id") == prov_id:
                ev["global_id"] = target_id

        if (prov_id, cam_id) in self._active_stopped_events:
            ev = self._active_stopped_events.pop((prov_id, cam_id))
            ev["global_id"] = target_id
            self._active_stopped_events[(target_id, cam_id)] = ev

        self.identities.pop(prov_id, None)
        if self.next_global_id == prov_id + 1:
            self.next_global_id = prov_id

        if mudou_de_camera:
            gap = max(0.0, entry_ts - prev_seen)
            msg = (
                f"Pessoa ID {target_id} moveu-se da câmara '{old_cam}' para '{cam_id}' "
                f"após {gap:.2f}s (distância Re-ID={reid_distance:.3f})"
            )
            logging.info(f"Re-ID Evento (Consolidado ID {prov_id} -> ID {target_id}): {msg}")
            self.event_log.append(
                {
                    "event_type": "CAMERA_TRANSITION",
                    "global_id": target_id,
                    "camera_id": cam_id,
                    "origin_camera": old_cam,
                    "next_camera": cam_id,
                    "timestamp_in": prev_seen,
                    "timestamp_out": entry_ts,
                    "duration_seconds": gap,
                    "stopped_seconds": 0.0,
                    "reid_distance": float(reid_distance),
                    "details": msg,
                }
            )
        return target_id

    def update_existing_identity(
        self,
        global_id: int,
        new_feature_vector: Optional[np.ndarray],
        bbox: list[float],
        cam_id: str,
        current_time: float,
        active_global_ids: Optional[set[int]] = None,
    ) -> Optional[int]:
        if active_global_ids is None:
            active_global_ids = set()

        with self._lock:
            self.cluster_manager.find(cam_id)
            identity = self.identities.get(global_id)
            if identity is None:
                return None

            is_edge = self._is_near_edge(bbox)
            feat_to_update = None if (new_feature_vector is None or is_edge) else new_feature_vector

            # 1. Online Provisional Identity Consolidation:
            # If this identity was recently created on border entry and now has a clean interior feature,
            # check if it actually belongs to an existing older identity (e.g. from another camera).
            if (
                feat_to_update is not None
                and global_id > 1
                and (current_time - identity.first_seen) <= 10.0
                and len(identity.passage_records) == 1
                and identity.passage_records[0].get("origin_camera") == "Entrada"
            ):
                exclude_ids = set(active_global_ids) | {global_id}
                match_id, _, raw_dist = self._find_best_matching_identity(
                    new_feature_vector=feat_to_update,
                    bbox=bbox,
                    cam_id=cam_id,
                    current_time=current_time,
                    active_global_ids=exclude_ids,
                    max_candidate_id=global_id,
                )
                if match_id is not None:
                    return self._merge_provisional_into_existing(
                        prov_id=global_id,
                        target_id=match_id,
                        new_feature_vector=feat_to_update,
                        bbox=bbox,
                        cam_id=cam_id,
                        current_time=current_time,
                        reid_distance=raw_dist,
                    )

            # 2. Cross-Camera Hijack Protection:
            # If this identity already moved to another camera (`identity.current_camera != cam_id`),
            # verify that the local track in `cam_id` still matches this person's appearance before
            # allowing it to pull the identity back or contaminate the gallery.
            prev_cam = identity.current_camera
            if prev_cam != cam_id and new_feature_vector is not None:
                verify_dist = identity.match_score(new_feature_vector, cam_id=cam_id)
                if verify_dist > (self.inter_camera_threshold * 1.08):
                    return None

            prev_seen = identity.last_seen_per_camera.get(prev_cam, identity.last_seen)
            switched = identity.update(
                feat_to_update,
                bbox,
                cam_id,
                current_time,
                switch_cooldown=self.switch_cooldown,
            )
            if switched and prev_cam != cam_id:
                self._check_and_update_clusters(identity, cam_id, current_time, reid_distance=0.0)
                gap = max(0.0, current_time - prev_seen)
                self.event_log.append(
                    {
                        "event_type": "CAMERA_TRANSITION",
                        "global_id": global_id,
                        "camera_id": cam_id,
                        "origin_camera": prev_cam,
                        "next_camera": cam_id,
                        "timestamp_in": prev_seen,
                        "timestamp_out": current_time,
                        "duration_seconds": gap,
                        "stopped_seconds": 0.0,
                        "reid_distance": 0.0,
                        "details": (
                            f"Pessoa ID {global_id} transitou da câmara '{prev_cam}' para '{cam_id}' "
                            f"(intervalo={gap:.2f}s)"
                        ),
                    }
                )
            return global_id

    def get_identity_current_camera(self, global_id: int) -> Optional[str]:
        with self._lock:
            ident = self.identities.get(global_id)
            return ident.current_camera if ident is not None else None

    def record_stopped_state(
        self,
        global_id: int,
        cam_id: str,
        is_stopped: bool,
        elapsed: float,
        current_time: float,
    ) -> None:
        """Record stopped/loitering behavior on the active identity and event log."""
        with self._lock:
            identity = self.identities.get(global_id)
            if identity is not None and identity.current_camera == cam_id:
                identity.last_seen = max(identity.last_seen, current_time)
                identity.last_seen_per_camera[cam_id] = max(
                    identity.last_seen_per_camera.get(cam_id, current_time), current_time
                )
                if identity.passage_records and identity.passage_records[-1]["camera_id"] == cam_id:
                    identity.passage_records[-1]["timestamp_out"] = max(
                        identity.passage_records[-1]["timestamp_out"], current_time
                    )
                if is_stopped and elapsed > 0.0:
                    identity.update_stopped_duration(cam_id, elapsed, current_time)

            key = (global_id, cam_id)
            if is_stopped and elapsed > 0.0 and (identity is None or identity.current_camera == cam_id):
                t_start = current_time - elapsed
                if key not in self._active_stopped_events:
                    ev = {
                        "event_type": "PERSON_STOPPED",
                        "global_id": global_id,
                        "camera_id": cam_id,
                        "origin_camera": cam_id,
                        "next_camera": cam_id,
                        "timestamp_in": t_start,
                        "timestamp_out": current_time,
                        "duration_seconds": elapsed,
                        "stopped_seconds": elapsed,
                        "reid_distance": 0.0,
                        "details": f"Pessoa ID {global_id} ficou parada na câmara '{cam_id}' por {elapsed:.1f}s",
                    }
                    self._active_stopped_events[key] = ev
                    self.event_log.append(ev)
                else:
                    ev = self._active_stopped_events[key]
                    ev["timestamp_out"] = current_time
                    ev["duration_seconds"] = elapsed
                    ev["stopped_seconds"] = elapsed
                    ev["details"] = f"Pessoa ID {global_id} ficou parada na câmara '{cam_id}' por {elapsed:.1f}s"
            else:
                self._active_stopped_events.pop(key, None)

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

        with self._lock:
            self.cluster_manager.find(cam_id)
            self._cleanup_old_identities(current_time)

            best_match_id, _, best_raw_app_dist = self._find_best_matching_identity(
                new_feature_vector=new_feature_vector,
                bbox=bbox,
                cam_id=cam_id,
                current_time=current_time,
                active_global_ids=active_global_ids,
            )

            is_edge = self._is_near_edge(bbox)
            if best_match_id is not None:
                identity = self.identities[best_match_id]
                old_cam = identity.current_camera
                prev_seen = identity.last_seen_per_camera.get(old_cam, identity.last_seen)

                if old_cam != cam_id:
                    self._check_and_update_clusters(
                        identity, cam_id, current_time, reid_distance=best_raw_app_dist
                    )

                feat_to_update = None if is_edge else new_feature_vector
                mudou_de_camera = identity.update(
                    feat_to_update,
                    bbox,
                    cam_id,
                    current_time,
                    switch_cooldown=self.switch_cooldown,
                    reid_distance=best_raw_app_dist,
                    force_transition=(old_cam != cam_id),
                )
                if mudou_de_camera:
                    gap = max(0.0, current_time - prev_seen)
                    msg = (
                        f"Pessoa ID {best_match_id} moveu-se da câmara '{old_cam}' para '{cam_id}' "
                        f"após {gap:.2f}s (distância Re-ID={best_raw_app_dist:.3f})"
                    )
                    logging.info(f"Re-ID Evento: {msg}")
                    self.event_log.append(
                        {
                            "event_type": "CAMERA_TRANSITION",
                            "global_id": best_match_id,
                            "camera_id": cam_id,
                            "origin_camera": old_cam,
                            "next_camera": cam_id,
                            "timestamp_in": prev_seen,
                            "timestamp_out": current_time,
                            "duration_seconds": gap,
                            "stopped_seconds": 0.0,
                            "reid_distance": float(best_raw_app_dist),
                            "details": msg,
                        }
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
            msg = f"Nova pessoa detectada -> Criado ID {new_id} na câmara '{cam_id}'"
            logging.info(f"Re-ID Evento: {msg}")
            self.event_log.append(
                {
                    "event_type": "PERSON_DETECTED",
                    "global_id": new_id,
                    "camera_id": cam_id,
                    "origin_camera": "Entrada",
                    "next_camera": cam_id,
                    "timestamp_in": current_time,
                    "timestamp_out": current_time,
                    "duration_seconds": 0.0,
                    "stopped_seconds": 0.0,
                    "reid_distance": 0.0,
                    "details": msg,
                }
            )
            return new_id

    def _aggregate_history_by_clusters(self, global_id: int, raw_history: list) -> list[dict]:
        if not raw_history:
            return []

        aggregated = []
        first_cam = raw_history[0]["camera_id"]
        current_cluster = self.cluster_manager.find(first_cam)
        t_in = raw_history[0]["timestamp_in"]
        t_out = raw_history[0]["timestamp_out"]
        stopped_sum = float(raw_history[0].get("stopped_seconds", 0.0))
        cams_in_stay = [first_cam]
        origin_env = "Entrada"

        for record in raw_history[1:]:
            cam = record["camera_id"]
            cluster = self.cluster_manager.find(cam)
            r_in = record["timestamp_in"]
            r_out = record["timestamp_out"]
            r_stop = float(record.get("stopped_seconds", 0.0))

            if cluster == current_cluster:
                t_out = max(t_out, r_out)
                stopped_sum += r_stop
                if cams_in_stay[-1] != cam:
                    cams_in_stay.append(cam)
            else:
                next_env = f"Ambiente_{cluster}"
                merged_label = self.cluster_manager.get_cluster_label(current_cluster)
                path_str = " -> ".join(cams_in_stay)
                dur = max(0.0, t_out - t_in)
                aggregated.append(
                    {
                        "event_type": "AMBIENTE_SUMMARY",
                        "global_id": global_id,
                        "camera_id": current_cluster,
                        "origin_camera": origin_env,
                        "next_camera": next_env,
                        "ambiente_id": f"Ambiente_{current_cluster}",
                        "merged_cameras": merged_label,
                        "cameras_path": path_str,
                        "timestamp_in": t_in,
                        "timestamp_out": t_out,
                        "duration_seconds": dur,
                        "stopped_seconds": stopped_sum,
                        "reid_distance": 0.0,
                        "details": (
                            f"Resumo Ambiente: Pessoa ID {global_id} permaneceu em 'Ambiente_{current_cluster}' "
                            f"(câmaras fundidas=[{merged_label}], percurso=[{path_str}]) por {dur:.2f}s"
                        ),
                    }
                )
                origin_env = f"Ambiente_{current_cluster}"
                current_cluster = cluster
                t_in = r_in
                t_out = r_out
                stopped_sum = r_stop
                cams_in_stay = [cam]

        merged_label = self.cluster_manager.get_cluster_label(current_cluster)
        path_str = " -> ".join(cams_in_stay)
        dur = max(0.0, t_out - t_in)
        aggregated.append(
            {
                "event_type": "AMBIENTE_SUMMARY",
                "global_id": global_id,
                "camera_id": current_cluster,
                "origin_camera": origin_env,
                "next_camera": "Saida",
                "ambiente_id": f"Ambiente_{current_cluster}",
                "merged_cameras": merged_label,
                "cameras_path": path_str,
                "timestamp_in": t_in,
                "timestamp_out": t_out,
                "duration_seconds": dur,
                "stopped_seconds": stopped_sum,
                "reid_distance": 0.0,
                "details": (
                    f"Resumo Ambiente: Pessoa ID {global_id} permaneceu em 'Ambiente_{current_cluster}' "
                    f"(câmaras fundidas=[{merged_label}], percurso=[{path_str}]) por {dur:.2f}s"
                ),
            }
        )
        return aggregated

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def get_identity_history(self, global_id: int) -> list[dict]:
        with self._lock:
            identity = self.identities.get(global_id) or self.archived_identities.get(global_id)
            if not identity:
                return []
            raw = identity.get_raw_history()

        clustered = self._aggregate_history_by_clusters(global_id, raw)

        formatted = []
        for record in clustered:
            formatted.append(
                {
                    "Ambiente": record["ambiente_id"],
                    "Cameras_Fundidas": record["merged_cameras"],
                    "Percurso_Cameras": record["cameras_path"],
                    "Entrada": self._format_timestamp(record["timestamp_in"]),
                    "Saida": self._format_timestamp(record["timestamp_out"]),
                    "Duracao_s": round(record["duration_seconds"], 2),
                }
            )
        return formatted

    def export_data_to_csv(self, filename: str = "tracking_data_final.csv") -> None:
        """
        Export a comprehensive, human- and machine-readable CSV log containing EVERYTHING that happened:
        - PERSON_DETECTED: Initial detection of each person in a camera.
        - CAMERA_PASSAGE: Every individual camera passage where each person passed by (preserving the
          exact physical camera_id, origin_camera, next_camera, full trajectory cameras_path, stopped time,
          AND the Ambiente / merged cameras group).
        - CAMERA_TRANSITION: Every cross-camera Re-ID movement with transition time and cosine distance.
        - CAMERA_MERGE: Every automatic camera fusion event into an Ambiente.
        - PERSON_STOPPED: Every loitering/stopped episode with duration.
        - AMBIENTE_SUMMARY: Consolidated stay per Ambiente cluster with full camera path.
        """
        with self._lock:
            all_identities: dict[int, GlobalIdentity] = {}
            all_identities.update(self.archived_identities)
            all_identities.update(self.identities)
            events_copy = [dict(ev) for ev in self.event_log]

        if not all_identities and not events_copy:
            return

        rows: list[dict] = []

        # 1. Add all chronological system events (PERSON_DETECTED, CAMERA_TRANSITION, CAMERA_MERGE, PERSON_STOPPED)
        for ev in events_copy:
            cam_id = ev.get("camera_id", "")
            root_cluster = self.cluster_manager.find(cam_id) if cam_id else ""
            merged_label = self.cluster_manager.get_cluster_label(cam_id) if cam_id else ""
            gid = ev.get("global_id", "")
            ident = all_identities.get(gid)
            full_path = (
                " -> ".join(r["camera_id"] for r in ident.get_raw_history())
                if ident is not None
                else ev.get("cameras_path", cam_id)
            )
            rows.append(
                {
                    "event_type": ev["event_type"],
                    "global_id": gid,
                    "camera_id": cam_id,
                    "origin_camera": ev.get("origin_camera", ""),
                    "next_camera": ev.get("next_camera", ""),
                    "ambiente_id": ev.get("ambiente_id") or (f"Ambiente_{root_cluster}" if root_cluster else ""),
                    "merged_cameras": ev.get("merged_cameras") or merged_label,
                    "cameras_path": ev.get("cameras_path") or full_path,
                    "timestamp_in": float(ev["timestamp_in"]),
                    "timestamp_out": float(ev["timestamp_out"]),
                    "duration_seconds": float(ev.get("duration_seconds", 0.0)),
                    "stopped_seconds": float(ev.get("stopped_seconds", 0.0)),
                    "reid_distance": float(ev.get("reid_distance", 0.0)),
                    "details": ev.get("details", ""),
                }
            )

        # 2. Add detailed per-camera passages (CAMERA_PASSAGE) and environment summaries (AMBIENTE_SUMMARY)
        for gid in sorted(all_identities.keys()):
            identity = all_identities[gid]
            raw_history = identity.get_raw_history()
            if not raw_history:
                continue

            full_path = " -> ".join(r["camera_id"] for r in raw_history)

            for rec in raw_history:
                cam_id = rec["camera_id"]
                root_cluster = self.cluster_manager.find(cam_id)
                merged_label = self.cluster_manager.get_cluster_label(cam_id)
                dur = float(rec["duration_seconds"])
                stop_s = float(rec["stopped_seconds"])
                orig = rec["origin_camera"]
                nxt = rec["next_camera"]
                reid_d = float(rec["reid_distance"])

                stop_note = f", parado={stop_s:.1f}s" if stop_s > 0 else ""
                reid_note = f", dist_reid={reid_d:.3f}" if reid_d > 0 else ""
                details = (
                    f"Passagem Câmara: Pessoa ID {gid} passou por '{cam_id}' "
                    f"(Ambiente_{root_cluster} [{merged_label}]) | Origem: {orig} -> Destino: {nxt} "
                    f"| Duração: {dur:.2f}s{stop_note}{reid_note} | Trajeto completo: [{full_path}]"
                )

                rows.append(
                    {
                        "event_type": "CAMERA_PASSAGE",
                        "global_id": gid,
                        "camera_id": cam_id,
                        "origin_camera": orig,
                        "next_camera": nxt,
                        "ambiente_id": f"Ambiente_{root_cluster}",
                        "merged_cameras": merged_label,
                        "cameras_path": full_path,
                        "timestamp_in": float(rec["timestamp_in"]),
                        "timestamp_out": float(rec["timestamp_out"]),
                        "duration_seconds": dur,
                        "stopped_seconds": stop_s,
                        "reid_distance": reid_d,
                        "details": details,
                    }
                )

            clustered_summaries = self._aggregate_history_by_clusters(gid, raw_history)
            rows.extend(clustered_summaries)

        # Sort chronological events/passages first by timestamp_in, followed by final AMBIENTE_SUMMARY rows
        type_priority = {
            "PERSON_DETECTED": 0,
            "CAMERA_TRANSITION": 1,
            "CAMERA_MERGE": 2,
            "CAMERA_PASSAGE": 3,
            "PERSON_STOPPED": 4,
            "AMBIENTE_SUMMARY": 5,
        }
        rows.sort(
            key=lambda r: (
                1 if r["event_type"] == "AMBIENTE_SUMMARY" else 0,
                round(float(r["timestamp_in"]), 3),
                type_priority.get(r["event_type"], 9),
                int(r["global_id"]) if str(r["global_id"]).isdigit() else 0,
            )
        )

        formatted_rows = []
        for idx, r in enumerate(rows, start=1):
            t_in = float(r["timestamp_in"])
            t_out = float(r["timestamp_out"])
            formatted_rows.append(
                {
                    "event_index": idx,
                    "event_type": r["event_type"],
                    "global_id": r["global_id"],
                    "camera_id": r["camera_id"],
                    "origin_camera": r["origin_camera"],
                    "next_camera": r["next_camera"],
                    "ambiente_id": r["ambiente_id"],
                    "merged_cameras": r["merged_cameras"],
                    "cameras_path": r["cameras_path"],
                    "timestamp_in": f"{t_in:.3f}",
                    "timestamp_out": f"{t_out:.3f}",
                    "datetime_in": self._format_timestamp(t_in),
                    "datetime_out": self._format_timestamp(t_out),
                    "duration_seconds": f"{float(r['duration_seconds']):.3f}",
                    "stopped_seconds": f"{float(r['stopped_seconds']):.3f}",
                    "reid_distance": f"{float(r['reid_distance']):.4f}",
                    "details": r["details"],
                }
            )

        with open(filename, "w", newline="", encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=self.CSV_FIELDS)
            dict_writer.writeheader()
            dict_writer.writerows(formatted_rows)
