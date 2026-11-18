# vision/cameraWorker.py
"""
CameraWorker: Per-camera tracking with `deep-sort-realtime`, global identity resolution,
decoupled behavior evaluation (`BehaviorPipeline`), and HUD annotation.
"""
import time
from typing import Optional
import numpy as np
import torch
from torchvision.ops import box_iou
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import TrackState

from behaviors.BehaviorPipeline import BehaviorPipeline
from domain.AnnotationSpec import AnnotationSpec
from domain.TrackContext import TrackContext
from ui.display import draw_annotation_spec, draw_camera_hud
from vision.StabilizedTrack import StabilizedTrack


class CameraWorker:
    """Per-camera tracker using `deep-sort-realtime` with `GlobalIdentityManager` integration."""

    def __init__(self, cam_id: str, config, global_manager, inference_engine=None, analyzers=None):
        self.cam_id = cam_id
        self.config = config
        self.global_manager = global_manager
        self._inference_engine = inference_engine
        self.local_to_global_map: dict[int, int] = {}
        self._last_annotations: list[AnnotationSpec] = []
        self.behavior_pipeline = BehaviorPipeline(self.config, analyzers=analyzers)
        self.tracker = self._create_tracker(config)
        self.conf_high = getattr(config, "YOLO_CONF_THRESHOLD", 0.40)
        self.max_coast_frames = getattr(config, "DEEPSORT_MAX_COAST_FRAMES", 3)
        self.feature_update_interval = getattr(config, "FEATURE_UPDATE_INTERVAL", 3)
        self.frame_w = getattr(config, "PROCESSING_WIDTH", 640)
        self.frame_h = getattr(config, "PROCESSING_HEIGHT", 480)
        self.frame_counter = 0
        self.prev_time = time.perf_counter()
        self.fps = 0.0
        print(f"[Worker-{self.cam_id}] Inicializado com deep-sort-realtime.")

    @staticmethod
    def _create_tracker(config) -> DeepSort:
        ds = DeepSort(
            max_age=getattr(config, "DEEPSORT_MAX_AGE", 90),
            n_init=getattr(config, "DEEPSORT_N_INIT", 2),
            nms_max_overlap=getattr(config, "DEEPSORT_NMS_MAX_OVERLAP", 0.75),
            max_cosine_distance=getattr(config, "DEEPSORT_MAX_COSINE_DISTANCE", 0.40),
            max_iou_distance=getattr(config, "DEEPSORT_MAX_IOU_DISTANCE", 0.72),
            nn_budget=getattr(config, "DEEPSORT_NN_BUDGET", 35),
            gating_only_position=True,
            override_track_class=StabilizedTrack,
            embedder=None,
        )
        ds.tracker.kf._std_weight_position = 1.0 / 12.0
        ds.tracker.kf._std_weight_velocity = 1.0 / 80.0
        return ds

    def _get_or_init_engine(self):
        if self._inference_engine is None:
            from vision.BatchInferenceEngine import BatchInferenceEngine
            self._inference_engine = BatchInferenceEngine(self.config)
        return self._inference_engine

    def _is_exiting_border(self, box: list[float]) -> bool:
        x1, y1, x2, y2 = box
        m = 6.0
        return x1 <= m or y1 <= m or x2 >= (self.frame_w - m) or y2 >= (self.frame_h - m)

    def _update_fps(self, current_time: Optional[float], override_fps: Optional[float]) -> float:
        now_perf = time.perf_counter()
        dt = now_perf - self.prev_time
        if override_fps is not None:
            self.fps = override_fps
        elif dt > 0:
            inst_fps = 1.0 / dt
            self.fps = 0.90 * self.fps + 0.10 * inst_fps if self.fps > 0 else inst_fps
        self.prev_time = now_perf
        return current_time if current_time is not None else time.time()

    def _render_cached_frame(self, frame: np.ndarray) -> np.ndarray:
        for spec in self._last_annotations:
            draw_annotation_spec(frame, spec, self.config)
        draw_camera_hud(frame, self.cam_id, self.fps, len(self._last_annotations))
        return frame

    @staticmethod
    def _compute_isolation_mask(boxes_xyxy: np.ndarray, thresh: float = 0.20) -> list[bool]:
        if len(boxes_xyxy) <= 1:
            return [True] * len(boxes_xyxy)
        iou_mat = box_iou(torch.as_tensor(boxes_xyxy, dtype=torch.float32), torch.as_tensor(boxes_xyxy, dtype=torch.float32))
        iou_mat.fill_diagonal_(0.0)
        return (iou_mat.amax(dim=1) < thresh).tolist()

    def _update_tracker(self, boxes_xyxy: np.ndarray, confs: np.ndarray, feats: np.ndarray) -> list:
        keep = confs >= self.conf_high
        boxes, scores, embeds = boxes_xyxy[keep], confs[keep], feats[keep]
        isolated = self._compute_isolation_mask(boxes, thresh=0.20)
        raw_dets = [
            ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(c), "person")
            for (x1, y1, x2, y2), c in zip(boxes, scores)
        ]
        tracks = self.tracker.update_tracks(raw_dets, embeds=embeds, others=isolated)
        self._suppress_duplicate_tracks(tracks)
        return tracks

    def _suppress_duplicate_tracks(self, tracks: list) -> None:
        alive = [t for t in tracks if not t.is_deleted()]
        if len(alive) <= 1:
            return
        alive.sort(key=lambda t: (0 if t.is_confirmed() else 1, t.time_since_update, -t.hits, int(t.track_id)))
        iou_mat = box_iou(torch.tensor([list(t.to_ltrb()) for t in alive], dtype=torch.float32), torch.tensor([list(t.to_ltrb()) for t in alive], dtype=torch.float32)).numpy()
        for i, ti in enumerate(alive):
            if ti.is_deleted():
                continue
            for j in range(i + 1, len(alive)):
                if not alive[j].is_deleted() and self._is_duplicate_pair(ti, alive[j], iou_mat[i, j]):
                    alive[j].state = TrackState.Deleted

    @staticmethod
    def _is_duplicate_pair(ti, tj, iou_val: float) -> bool:
        sim = float(np.dot(ti.get_feature(), tj.get_feature()))
        thresh = 0.70 if (ti.time_since_update == 0 and tj.time_since_update == 0) else 0.52
        return iou_val > thresh and sim > 0.74

    def _retire_migrated_tracks(self, tracks: list) -> None:
        if not hasattr(self.global_manager, "get_identity_current_camera"):
            return
        for tr in tracks:
            lid = int(tr.track_id)
            if not tr.is_deleted() and tr.time_since_update > 15 and lid in self.local_to_global_map:
                owner = self.global_manager.get_identity_current_camera(self.local_to_global_map[lid])
                if owner is not None and owner != self.cam_id:
                    del self.local_to_global_map[lid]
                    tr.state = TrackState.Deleted

    def _cleanup_local_map(self, tracks: list) -> None:
        alive_ids = {int(t.track_id) for t in tracks if not t.is_deleted()}
        for lid in [k for k in self.local_to_global_map if k not in alive_ids]:
            del self.local_to_global_map[lid]

    def _get_active_global_ids(self, tracks: list) -> set[int]:
        return {
            self.local_to_global_map[int(t.track_id)]
            for t in tracks
            if not t.is_deleted() and int(t.track_id) in self.local_to_global_map and t.time_since_update == 0
        }

    def _get_sorted_candidates(self, tracks: list) -> list:
        candidates = [t for t in tracks if t.is_confirmed() and not t.is_deleted()]
        candidates.sort(
            key=lambda t: (0 if t.time_since_update == 0 else 1, 0 if int(t.track_id) in self.local_to_global_map else 1, -t.hits, int(t.track_id))
        )
        return candidates

    def _is_renderable_track(self, track, box: list[float]) -> bool:
        if track.time_since_update == 0:
            return True
        return (
            track.time_since_update <= self.max_coast_frames
            and track.hits >= 3
            and not self._is_exiting_border(box)
        )

    def _retire_replaced_tracks(self, tracks: list, active_lid: int, global_id: int) -> None:
        for other_t in tracks:
            other_lid = int(other_t.track_id)
            if other_lid != active_lid and self.local_to_global_map.get(other_lid) == global_id:
                del self.local_to_global_map[other_lid]
                if other_t.time_since_update > 0:
                    other_t.state = TrackState.Deleted

    def _assign_new_global_id(
        self, tracks: list, local_id: int, feat: np.ndarray, box: list[float], current_time: float, blocked_ids: set[int]
    ) -> int:
        gid = self.global_manager.get_or_create_global_id(feat, box, self.cam_id, current_time, blocked_ids)
        self._retire_replaced_tracks(tracks, local_id, gid)
        self.local_to_global_map[local_id] = gid
        return gid

    def _sync_existing_global_id(
        self, tracks: list, track, box: list[float], is_isolated: bool, current_time: float, blocked_ids: set[int]
    ) -> int:
        local_id = int(track.track_id)
        gid = self.local_to_global_map[local_id]
        do_sync = (self.frame_counter % self.feature_update_interval == 0) or (track.hits <= 18)
        if track.time_since_update > 0 or not do_sync:
            return gid
        feat = track.get_feature() if is_isolated else None
        resolved = self.global_manager.update_existing_identity(gid, feat, box, self.cam_id, current_time, blocked_ids)
        return self._handle_resolved_id(tracks, local_id, gid, resolved, track.get_feature(), box, current_time, blocked_ids)

    def _handle_resolved_id(
        self, tracks: list, local_id: int, old_gid: int, resolved: Optional[int], feat: np.ndarray, box: list[float], current_time: float, blocked_ids: set[int]
    ) -> int:
        if resolved is None:
            return self._assign_new_global_id(tracks, local_id, feat, box, current_time, blocked_ids - {old_gid})
        if resolved != old_gid:
            self._retire_replaced_tracks(tracks, local_id, resolved)
            self.local_to_global_map[local_id] = resolved
        return resolved

    def _resolve_track_global_id(
        self, tracks: list, track, box: list[float], is_isolated: bool, current_time: float, blocked_ids: set[int]
    ) -> int:
        local_id = int(track.track_id)
        if local_id not in self.local_to_global_map:
            return self._assign_new_global_id(tracks, local_id, track.get_feature(), box, current_time, blocked_ids)
        return self._sync_existing_global_id(tracks, track, box, is_isolated, current_time, blocked_ids)

    def _evaluate_and_draw_track(self, frame: np.ndarray, track, global_id: int, box: list[float], current_time: float) -> AnnotationSpec:
        do_sync = (self.frame_counter % self.feature_update_interval == 0)
        ctx = TrackContext(
            global_id=global_id,
            local_id=int(track.track_id),
            cam_id=self.cam_id,
            bbox=box,
            current_time=current_time,
            confidence=float(track.det_conf or 1.0),
        )
        spec = self.behavior_pipeline.evaluate(ctx, global_manager=self.global_manager, force_sync=do_sync)
        draw_annotation_spec(frame, spec, self.config)
        return spec

    def _render_active_tracks(self, frame: np.ndarray, tracks: list, current_time: float) -> None:
        candidates = self._get_sorted_candidates(tracks)
        cand_boxes = np.array([list(t.to_ltrb()) for t in candidates], dtype=np.float32) if candidates else np.empty((0, 4))
        cand_isolated = self._compute_isolation_mask(cand_boxes, thresh=0.18)
        active_gids = self._get_active_global_ids(tracks)
        rendered_gids: set[int] = set()
        annotations: list[AnnotationSpec] = []

        for idx, track in enumerate(candidates):
            box = [float(v) for v in track.to_ltrb()]
            if not self._is_renderable_track(track, box):
                continue
            gid = self._resolve_track_global_id(tracks, track, box, cand_isolated[idx], current_time, active_gids | rendered_gids)
            active_gids.add(gid)
            if gid not in rendered_gids:
                rendered_gids.add(gid)
                annotations.append(self._evaluate_and_draw_track(frame, track, gid, box, current_time))
        self._last_annotations = annotations

    def process_detections(
        self, frame: np.ndarray, boxes_xyxy: np.ndarray, confs: np.ndarray,
        feats: np.ndarray, current_time: Optional[float] = None,
        override_fps: Optional[float] = None, is_new_frame: bool = True,
    ) -> np.ndarray:
        current_time = self._update_fps(current_time, override_fps)
        if not is_new_frame:
            return self._render_cached_frame(frame)

        self.frame_counter += 1
        tracks = self._update_tracker(boxes_xyxy, confs, feats)
        self._retire_migrated_tracks(tracks)
        self._cleanup_local_map(tracks)
        self._render_active_tracks(frame, tracks, current_time)
        draw_camera_hud(frame, self.cam_id, self.fps, len(self._last_annotations))
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        boxes_xyxy, confs, feats = self._get_or_init_engine().infer_batch([frame])[0]
        return self.process_detections(frame, boxes_xyxy, confs, feats)

    def cleanup(self) -> None:
        print(f"[Worker-{self.cam_id}] Recursos libertados.")
