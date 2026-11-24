# identity/GlobalIdentityManager.py
import logging
import threading
from typing import Optional
import numpy as np

from domain.GlobalIdentity import GlobalIdentity
from exporters.ExportPipeline import ExportPipeline
from identity.CameraClusterManager import CameraClusterManager
from interfaces.IIdentityRepository import IIdentityRepository


class GlobalIdentityManager(IIdentityRepository):
    """Thread-safe repository managing cross-camera Re-ID matching and lifecycle (SRP & DIP)."""

    def __init__(self, config=None, exporters=None):
        self.config = config
        self.intra_camera_threshold = getattr(config, "SIMILARITY_THRESHOLD", 0.35)
        self.inter_camera_threshold = getattr(config, "INTER_CAMERA_THRESHOLD", 0.37)
        self.simultaneous_inter_camera_threshold = getattr(config, "SIMULTANEOUS_INTER_CAMERA_THRESHOLD", 0.30)
        self.max_time_lost = getattr(config, "MAX_TIME_LOST", 120.0)
        self.max_spatial_distance = float(getattr(config, "MAX_SPATIAL_DISTANCE", 180))
        self.switch_cooldown = getattr(config, "CAMERA_SWITCH_COOLDOWN", 0.8)
        self.auto_cluster_threshold = getattr(config, "AUTO_CLUSTER_TIME_THRESHOLD", 15.0)
        self.frame_w, self.frame_h = getattr(config, "PROCESSING_WIDTH", 640), getattr(config, "PROCESSING_HEIGHT", 480)
        self.cluster_manager = CameraClusterManager()
        self.export_pipeline = ExportPipeline(self.cluster_manager, config=config, exporters=exporters)
        self.identities: dict[int, GlobalIdentity] = {}
        self.archived_identities: dict[int, GlobalIdentity] = {}
        self.event_log: list[dict] = []
        self._active_stopped_events: dict[tuple[int, str, str], dict] = {}
        self.camera_telemetry_cache: dict[str, dict] = {}
        self.identity_telemetry_cache: dict[int, dict] = {}
        self.next_global_id, self._last_cleanup_time = 1, 0.0
        self._lock = threading.RLock()

    @staticmethod
    def _center(bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

    def _is_near_edge(self, bbox: list[float]) -> bool:
        x1, y1, x2, y2 = bbox
        w, h, mx, my = x2 - x1, y2 - y1, self.frame_w * 0.03, self.frame_h * 0.03
        return (((x1 < mx) or (x2 > self.frame_w - mx)) and (w < self.frame_w * 0.25)) or (
            ((y1 < my) or (y2 > self.frame_h - my)) and (h < self.frame_h * 0.20)
        )

    def _cleanup_old_identities(self, current_time: float) -> None:
        if (current_time - self._last_cleanup_time) < 2.0:
            return
        self._last_cleanup_time = current_time
        expiry = self.max_time_lost * 3.0
        for gid in [g for g, ident in self.identities.items() if (current_time - ident.last_seen) > expiry]:
            self.archived_identities[gid] = self.identities.pop(gid)

    def _enrich_event_telemetry(self, row: dict, gid: int, cam_id: str, dist: float, telemetry: Optional[dict] = None) -> None:
        if telemetry:
            self.camera_telemetry_cache[cam_id], self.identity_telemetry_cache[gid] = telemetry, telemetry
        src = telemetry or self.identity_telemetry_cache.get(gid) or self.camera_telemetry_cache.get(cam_id) or {}
        ident, root = self.identities.get(gid), (self.cluster_manager.find(cam_id) if cam_id else "")
        merged = self.cluster_manager.get_cluster_label(cam_id) if cam_id else ""
        path = " -> ".join(r["camera_id"] for r in ident.get_raw_history()) if ident else row.get("cameras_path", cam_id)
        limiares = dict(src.get("limiares_percentuais", {}))
        reid_sim_pct = round(max(0.0, (1.0 - float(dist))) * 100.0, 2)
        limiares.update({"reid_similarity_pct": reid_sim_pct, "reid_distance_pct": round(float(dist) * 100.0, 2)})
        row.update({
            "kafka_topic": getattr(self.config, "KAFKA_EVENTS_TOPIC", "Calma_Nathalia"),
            "ambiente_id": row.get("ambiente_id") or (f"Ambiente_{root}" if root else ""),
            "merged_cameras": row.get("merged_cameras") or merged, "cameras_path": row.get("cameras_path") or path,
            "people_in_camera": int(src.get("people_in_camera", max(1, len(self.identities)))),
            "object_class": str(src.get("object_class", "person")), "class_id": int(src.get("class_id", 0)),
            "confidence_pct": float(src.get("confidence_pct", 100.0)), "reid_similarity_pct": reid_sim_pct,
            "limiares_percentuais": limiares, "camera_summary": dict(src.get("camera_summary", {})),
            "yolo": dict(src.get("yolo", {})), "deepsort": dict(src.get("deepsort", {})),
        })

    def _log_event(self, ev_type: str, gid: int, cam_id: str, origin: str, nxt: str, t_in: float, t_out: float, dist: float, details: str, **extra) -> None:
        row = {
            "event_type": ev_type, "global_id": gid, "camera_id": cam_id, "origin_camera": origin,
            "next_camera": nxt, "timestamp_in": t_in, "timestamp_out": t_out,
            "duration_seconds": max(0.0, t_out - t_in), "stopped_seconds": 0.0,
            "reid_distance": float(dist), "details": details,
        }
        row.update(extra)
        self._enrich_event_telemetry(row, gid, cam_id, dist)
        self.event_log.append(row)
        self.export_pipeline.notify_event(row)

    def _check_and_update_clusters(self, identity: GlobalIdentity, cam_id: str, current_time: float, reid_distance: float = 0.0) -> None:
        self.cluster_manager.find(cam_id)
        for old_cam, last_t in list(identity.last_seen_per_camera.items()):
            gap = current_time - last_t
            if old_cam != cam_id and 0.0 <= gap <= self.auto_cluster_threshold and self.cluster_manager.union(cam_id, old_cam):
                root, merged = self.cluster_manager.find(cam_id), self.cluster_manager.get_cluster_label(cam_id)
                msg = f"Câmaras '{old_cam}' e '{cam_id}' fundidas no 'Ambiente_{root}' ([{merged}]) via ID {identity.global_id}"
                logging.info(f"Mapeamento: {msg}")
                self._log_event("CAMERA_MERGE", identity.global_id, cam_id, old_cam, cam_id, last_t, current_time, reid_distance, msg, ambiente_id=f"Ambiente_{root}", merged_cameras=merged, cameras_path=f"{old_cam} -> {cam_id}")

    def _score_same_camera(self, ident: GlobalIdentity, cam_id: str, bbox: list[float], app_dist: float, time_lost: float) -> Optional[float]:
        ref = ident.last_bbox_per_camera.get(cam_id, ident.last_bbox)
        cx, cy = self._center(bbox)
        old_cx, old_cy = self._center(ref)
        dist = float(np.hypot(cx - old_cx, cy - old_cy))
        new_h, old_h = max(16.0, float(bbox[3] - bbox[1])), max(16.0, float(ref[3] - ref[1]))
        scale = min(new_h, old_h) / max(new_h, old_h)
        if time_lost < 3.0 and (dist > max(self.max_spatial_distance, 240.0 * max(0.45, time_lost)) or scale < 0.45):
            return None
        bonus = 0.065 if (time_lost < 2.5 and dist <= self.max_spatial_distance and scale >= 0.62) else 0.0
        eff = max(0.0, app_dist - bonus)
        return eff if eff < self.intra_camera_threshold else None

    def _score_cross_camera(self, ident: GlobalIdentity, cam_id: str, app_dist: float, current_time: float) -> Optional[float]:
        seen_primary = ident.last_seen_per_camera.get(ident.current_camera, ident.last_seen)
        dt_primary = current_time - seen_primary
        exiting = self._is_near_edge(ident.last_bbox)
        if dt_primary < 0.35 and not exiting:
            return app_dist if app_dist < self.simultaneous_inter_camera_threshold else None
        same_cluster = self.cluster_manager.find(ident.current_camera) == self.cluster_manager.find(cam_id)
        bonus = (0.15 <= dt_primary <= self.auto_cluster_threshold) or (same_cluster and dt_primary <= 25.0)
        eff = max(0.0, app_dist - 0.035) if bonus else app_dist
        return eff if eff < self.inter_camera_threshold else None

    def _find_best_matching_identity(self, new_feature_vector: np.ndarray, bbox: list[float], cam_id: str, current_time: float, active_global_ids: set[int], max_candidate_id: Optional[int] = None) -> tuple[Optional[int], float, float]:
        best_id, best_dist, best_raw = None, float("inf"), 0.0
        for gid, ident in self.identities.items():
            if (max_candidate_id is not None and gid >= max_candidate_id) or (gid in active_global_ids) or ((current_time - ident.last_seen) > self.max_time_lost):
                continue
            raw_dist = ident.match_score(new_feature_vector, cam_id=cam_id)
            eff = self._score_same_camera(ident, cam_id, bbox, raw_dist, current_time - ident.last_seen) if ident.current_camera == cam_id else self._score_cross_camera(ident, cam_id, raw_dist, current_time)
            if eff is not None and eff < best_dist:
                best_id, best_dist, best_raw = gid, eff, raw_dist
        return best_id, best_dist, best_raw

    def _rebind_provisional_logs(self, prov_id: int, target_id: int, cam_id: str) -> None:
        self.event_log = [ev for ev in self.event_log if not (ev.get("global_id") == prov_id and ev.get("event_type") == "PERSON_DETECTED")]
        for ev in self.event_log:
            if ev.get("global_id") == prov_id:
                ev["global_id"] = target_id
        for key in [k for k in list(self._active_stopped_events.keys()) if k[0] == prov_id and k[1] == cam_id]:
            ev = self._active_stopped_events.pop(key)
            ev["global_id"] = target_id
            self._active_stopped_events[(target_id, cam_id, key[2])] = ev

    def _merge_provisional_into_existing(self, prov_id: int, target_id: int, feat: Optional[np.ndarray], bbox: list[float], cam_id: str, current_time: float, dist: float) -> int:
        prov, target = self.identities.get(prov_id), self.identities.get(target_id)
        if prov is None or target is None:
            return target_id
        old_cam, prev_seen, entry_ts = target.current_camera, target.last_seen_per_camera.get(target.current_camera, target.last_seen), prov.first_seen
        if old_cam != cam_id:
            self._check_and_update_clusters(target, cam_id, entry_ts, reid_distance=dist)
        switched = target.update(feat, bbox, cam_id, current_time, self.switch_cooldown, dist, force_transition=(old_cam != cam_id))
        if target.passage_records and target.passage_records[-1]["camera_id"] == cam_id:
            target.passage_records[-1]["timestamp_in"] = min(target.passage_records[-1]["timestamp_in"], entry_ts)
        self._rebind_provisional_logs(prov_id, target_id, cam_id)
        self.identities.pop(prov_id, None)
        if self.next_global_id == prov_id + 1:
            self.next_global_id = prov_id
        if switched:
            self._log_event("CAMERA_TRANSITION", target_id, cam_id, old_cam, cam_id, prev_seen, entry_ts, dist, f"Pessoa ID {target_id} moveu-se de '{old_cam}' para '{cam_id}'")
        return target_id

    def _update_matched_identity(self, match_id: int, feat: np.ndarray, bbox: list[float], cam_id: str, current_time: float, raw_dist: float) -> int:
        ident = self.identities[match_id]
        old_cam, prev_seen = ident.current_camera, ident.last_seen_per_camera.get(ident.current_camera, ident.last_seen)
        if old_cam != cam_id:
            self._check_and_update_clusters(ident, cam_id, current_time, reid_distance=raw_dist)
        feat_upd = None if self._is_near_edge(bbox) else feat
        if ident.update(feat_upd, bbox, cam_id, current_time, self.switch_cooldown, raw_dist, force_transition=(old_cam != cam_id)):
            msg = f"Pessoa ID {match_id} moveu-se de '{old_cam}' para '{cam_id}'"
            logging.info(f"Re-ID Evento: {msg}")
            self._log_event("CAMERA_TRANSITION", match_id, cam_id, old_cam, cam_id, prev_seen, current_time, raw_dist, msg)
        return match_id

    def get_or_create_global_id(self, new_feature_vector: np.ndarray, bbox: list[float], cam_id: str, current_time: float, active_global_ids: Optional[set[int]] = None) -> int:
        with self._lock:
            self.cluster_manager.find(cam_id)
            self._cleanup_old_identities(current_time)
            match_id, _, raw_dist = self._find_best_matching_identity(new_feature_vector, bbox, cam_id, current_time, active_global_ids or set())
            if match_id is not None:
                return self._update_matched_identity(match_id, new_feature_vector, bbox, cam_id, current_time, raw_dist)
            new_id = self.next_global_id
            self.identities[new_id] = GlobalIdentity(new_id, new_feature_vector, bbox, cam_id, current_time)
            self.next_global_id += 1
            msg = f"Nova pessoa detectada -> Criado ID {new_id} na câmara '{cam_id}'"
            logging.info(f"Re-ID Evento: {msg}")
            self._log_event("PERSON_DETECTED", new_id, cam_id, "Entrada", cam_id, current_time, current_time, 0.0, msg)
            return new_id

    def _try_consolidate_provisional(self, gid: int, ident: GlobalIdentity, feat: Optional[np.ndarray], bbox: list[float], cam_id: str, current_time: float, active_ids: set[int]) -> Optional[int]:
        if feat is None or gid <= 1 or (current_time - ident.first_seen) > 15.0 or len(ident.passage_records) != 1 or ident.passage_records[0].get("origin_camera") != "Entrada":
            return None
        match_id, _, raw_dist = self._find_best_matching_identity(feat, bbox, cam_id, current_time, set(active_ids) | {gid}, max_candidate_id=gid)
        if match_id is None:
            return None
        return self._merge_provisional_into_existing(gid, match_id, feat, bbox, cam_id, current_time, raw_dist)

    def update_existing_identity(self, global_id: int, new_feature_vector: Optional[np.ndarray], bbox: list[float], cam_id: str, current_time: float, active_global_ids: Optional[set[int]] = None) -> Optional[int]:
        with self._lock:
            self.cluster_manager.find(cam_id)
            ident = self.identities.get(global_id)
            if ident is None:
                return None
            feat_upd = None if (new_feature_vector is None or self._is_near_edge(bbox)) else new_feature_vector
            merged_id = self._try_consolidate_provisional(global_id, ident, feat_upd, bbox, cam_id, current_time, active_global_ids or set())
            if merged_id is not None:
                return merged_id
            prev_cam, prev_seen = ident.current_camera, ident.last_seen_per_camera.get(ident.current_camera, ident.last_seen)
            if prev_cam != cam_id and new_feature_vector is not None and ident.match_score(new_feature_vector, cam_id=cam_id) > (self.inter_camera_threshold * 1.15):
                return global_id
            if ident.update(feat_upd, bbox, cam_id, current_time, switch_cooldown=self.switch_cooldown) and prev_cam != cam_id:
                self._check_and_update_clusters(ident, cam_id, current_time)
                self._log_event("CAMERA_TRANSITION", global_id, cam_id, prev_cam, cam_id, prev_seen, current_time, 0.0, f"Pessoa ID {global_id} transitou de '{prev_cam}' para '{cam_id}'")
            return global_id

    def get_identity_current_camera(self, global_id: int) -> Optional[str]:
        with self._lock:
            ident = self.identities.get(global_id)
            return ident.current_camera if ident is not None else None

    def _upsert_behavior_log(self, key: tuple[int, str, str], ev_type: str, behavior_name: str, duration: float, current_time: float, details: str, telemetry: Optional[dict] = None) -> None:
        gid, cam_id, _ = key
        ident = self.identities.get(gid)
        dist = float(ident.passage_records[-1].get("reid_distance", 0.0)) if (ident and ident.passage_records) else 0.0
        desc = details or f"Evento '{ev_type}' para ID {gid} em '{cam_id}' ({duration:.1f}s)"
        is_new = key not in self._active_stopped_events
        ev = self._active_stopped_events.get(key) or {
            "event_type": ev_type, "global_id": gid, "camera_id": cam_id, "origin_camera": cam_id, "next_camera": cam_id,
            "timestamp_in": current_time - duration, "timestamp_out": current_time, "duration_seconds": duration,
            "stopped_seconds": duration if behavior_name == "stopped" else 0.0, "reid_distance": dist, "details": desc,
        }
        ev.update({"timestamp_out": current_time, "duration_seconds": duration, "details": desc, "reid_distance": dist})
        if behavior_name == "stopped":
            ev["stopped_seconds"] = duration
        self._enrich_event_telemetry(ev, gid, cam_id, dist, telemetry)
        if is_new:
            self._active_stopped_events[key] = ev
            self.event_log.append(ev)
        self.export_pipeline.notify_event(ev)

    def record_behavior_event(self, global_id: int, cam_id: str, behavior_name: str, event_type: str, is_active: bool, duration_seconds: float, current_time: float, details: str = "", telemetry: Optional[dict] = None) -> None:
        with self._lock:
            ident = self.identities.get(global_id)
            if ident is not None and ident.current_camera == cam_id and behavior_name == "stopped" and is_active and duration_seconds > 0.0:
                ident.update_stopped_duration(cam_id, duration_seconds, current_time)
            key = (global_id, cam_id, behavior_name)
            valid_dur = (duration_seconds >= 0.0) if behavior_name != "stopped" else (duration_seconds > 0.0)
            if is_active and valid_dur and (ident is None or ident.current_camera == cam_id):
                self._upsert_behavior_log(key, event_type, behavior_name, duration_seconds, current_time, details, telemetry)
            else:
                self._active_stopped_events.pop(key, None)

    def record_stopped_state(self, global_id: int, cam_id: str, is_stopped: bool, elapsed: float, current_time: float) -> None:
        self.record_behavior_event(global_id, cam_id, "stopped", "PERSON_STOPPED", is_stopped, elapsed, current_time)

    def export_data(self) -> None:
        """Export consolidated tracking data through all registered `IDataExporter` sinks."""
        with self._lock:
            all_identities = {**self.archived_identities, **self.identities}
            events_copy = [dict(ev) for ev in self.event_log]
        self.export_pipeline.export_all(all_identities, events_copy)

    def export_data_to_csv(self, filename: str = "tracking_data_final.csv") -> None:
        self.export_data()
