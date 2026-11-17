# vision/cameraWorker.py
"""
CameraWorker: Per-camera tracking using the `deep-sort-realtime` library (`DeepSort` & `Track`),
cross-camera identity resolution (`GlobalIdentityManager`), stopped behavior evaluation
(`StoppedStateTracker`), and visual HUD annotation.
"""
import time
from typing import Optional
import numpy as np
import torch
from torchvision.ops import box_iou
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track, TrackState

from core.BehaviorPipeline import BehaviorPipeline
from core.domain import AnnotationSpec, TrackContext
from ui.display import draw_annotation_spec, draw_camera_hud


class StabilizedTrack(Track):
    """
    Subclass of `deep_sort_realtime.deep_sort.track.Track` (used via `override_track_class`)
    that adds:
    1. Velocity damping during occlusion so lost Kalman tracks never drift onto other people.
    2. EMA bounding-box smoothing (`smooth_ltrb`) to eliminate detector jitter.
    3. Outlier-filtered EMA appearance feature (`ema_feat`) so crossing pedestrians do not
       contaminate `deep-sort-realtime`'s appearance gallery.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        feat = self.features[-1] if self.features else np.zeros(576, dtype=np.float32)
        self.ema_feat = self._normalize(feat)
        self.smooth_ltrb = list(super().to_ltrb(orig=True))
        self.max_conf = float(self.det_conf or 0.0)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-6 else arr

    def predict(self, kf):
        # Zero aspect-ratio velocity and damp linear velocity during occlusion so boxes never drift
        self.mean[6] = 0.0
        if 1 <= self.time_since_update <= 5:
            self.mean[4:6] *= 0.60
            self.mean[7] = 0.0
        elif self.time_since_update > 5:
            self.mean[4:8] = 0.0

        super().predict(kf)
        self.mean[2] = float(np.clip(self.mean[2], 0.14, 2.85))
        self.mean[3] = max(16.0, float(self.mean[3]))

        if self.time_since_update in (1, 2):
            damp = 0.30 if self.time_since_update == 1 else 0.15
            vx, vy = float(self.mean[4]) * damp, float(self.mean[5]) * damp
            self.smooth_ltrb = [
                self.smooth_ltrb[0] + vx,
                self.smooth_ltrb[1] + vy,
                self.smooth_ltrb[2] + vx,
                self.smooth_ltrb[3] + vy,
            ]

    def update(self, kf, detection):
        prev_feat = self.ema_feat
        super().update(kf, detection)
        self.mean[6] = 0.0
        self.max_conf = max(self.max_conf, float(detection.confidence or 0.0))
        if self.state == TrackState.Confirmed and self.max_conf < 0.50:
            self.state = TrackState.Tentative

        # Smooth bounding box between detector measurement and previous stabilized box
        det_box = np.asarray(super().to_ltrb(orig=True), dtype=np.float64)
        kalman_box = np.asarray(super().to_ltrb(orig=False), dtype=np.float64)
        target = 0.72 * det_box + 0.28 * kalman_box
        prev_box = np.asarray(self.smooth_ltrb, dtype=np.float64)
        h = max(16.0, float(target[3] - target[1]))
        shift = float(np.hypot(
            0.5 * (target[0] + target[2] - prev_box[0] - prev_box[2]),
            0.5 * (target[1] + target[3] - prev_box[1] - prev_box[3]),
        ))
        alpha = min(0.92, 0.65 + 0.35 * min(1.0, shift / (0.12 * h)))
        self.smooth_ltrb = [float(x) for x in (alpha * target + (1.0 - alpha) * prev_box)]

        # Only update appearance gallery when detection is isolated (not occluded by another person)
        is_isolated = bool(detection.others) if detection.others is not None else True
        if is_isolated and detection.feature is not None:
            unit_f = self._normalize(detection.feature)
            sim = float(np.dot(prev_feat, unit_f))
            if self.hits <= 4 or sim >= 0.52:
                w = 0.55 if self.hits <= 4 else 0.84
                self.ema_feat = self._normalize(w * prev_feat + (1.0 - w) * unit_f)
                self.features[-1] = self.ema_feat
            else:
                self.features[-1] = prev_feat
        else:
            self.features[-1] = prev_feat

    def to_ltrb(self, orig=False, orig_strict=False):
        return np.asarray(self.smooth_ltrb, dtype=np.float32)

    def get_feature(self):
        return self.ema_feat


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

        self.tracker = DeepSort(
            max_age=getattr(config, "DEEPSORT_MAX_AGE", 75),
            n_init=getattr(config, "DEEPSORT_N_INIT", 2),
            nms_max_overlap=getattr(config, "DEEPSORT_NMS_MAX_OVERLAP", 0.85),
            max_cosine_distance=getattr(config, "DEEPSORT_MAX_COSINE_DISTANCE", 0.30),
            max_iou_distance=getattr(config, "DEEPSORT_MAX_IOU_DISTANCE", 0.62),
            nn_budget=getattr(config, "DEEPSORT_NN_BUDGET", 35),
            gating_only_position=True,
            override_track_class=StabilizedTrack,
            embedder=None,
        )

        self.conf_high = getattr(config, "YOLO_CONF_THRESHOLD", 0.28)
        self.max_coast_frames = getattr(config, "DEEPSORT_MAX_COAST_FRAMES", 2)
        self.feature_update_interval = getattr(config, "FEATURE_UPDATE_INTERVAL", 3)
        self.frame_w = getattr(config, "PROCESSING_WIDTH", 640)
        self.frame_h = getattr(config, "PROCESSING_HEIGHT", 480)
        self.frame_counter = 0

        self.prev_time = time.perf_counter()
        self.fps = 0.0

        print(f"[Worker-{self.cam_id}] Inicializado com deep-sort-realtime.")

    def _get_or_init_engine(self):
        if self._inference_engine is None:
            from vision.BatchInferenceEngine import BatchInferenceEngine
            self._inference_engine = BatchInferenceEngine(self.config)
        return self._inference_engine

    def _is_exiting_border(self, box: list[float]) -> bool:
        x1, y1, x2, y2 = box
        margin = 6.0
        return x1 <= margin or y1 <= margin or x2 >= (self.frame_w - margin) or y2 >= (self.frame_h - margin)

    def _suppress_duplicate_tracks(self, tracks: list) -> None:
        """Use `torchvision.ops.box_iou` to delete duplicate overlapping tracks of the same person."""
        alive = [t for t in tracks if not t.is_deleted()]
        if len(alive) <= 1:
            return
        alive.sort(key=lambda t: (0 if t.is_confirmed() else 1, t.time_since_update, -t.hits, int(t.track_id)))
        boxes_t = torch.tensor([list(t.to_ltrb()) for t in alive], dtype=torch.float32)
        iou_mat = box_iou(boxes_t, boxes_t).numpy()

        suppressed: set[int] = set()
        for i in range(len(alive)):
            if i in suppressed:
                continue
            ti = alive[i]
            for j in range(i + 1, len(alive)):
                if j in suppressed:
                    continue
                tj = alive[j]
                sim = float(np.dot(ti.get_feature(), tj.get_feature()))
                iou_thresh = 0.70 if (ti.time_since_update == 0 and tj.time_since_update == 0) else 0.52
                if iou_mat[i, j] > iou_thresh and sim > 0.74:
                    tj.state = TrackState.Deleted
                    suppressed.add(j)

    def process_detections(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        feats: np.ndarray,
        current_time: Optional[float] = None,
        override_fps: Optional[float] = None,
        is_new_frame: bool = True,
    ) -> np.ndarray:
        now_perf = time.perf_counter()
        if current_time is None:
            current_time = time.time()

        if override_fps is not None:
            self.fps = override_fps
        else:
            dt = now_perf - self.prev_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = 0.90 * self.fps + 0.10 * inst_fps if self.fps > 0 else inst_fps
        self.prev_time = now_perf

        if not is_new_frame:
            for spec in self._last_annotations:
                draw_annotation_spec(frame, spec, self.config)
            draw_camera_hud(frame, self.cam_id, self.fps, len(self._last_annotations))
            return frame

        self.frame_counter += 1

        # Filter confident detections and check isolation via Torchvision `box_iou`
        keep = confs >= self.conf_high
        boxes_xyxy, confs, feats = boxes_xyxy[keep], confs[keep], feats[keep]

        if len(boxes_xyxy) > 1:
            iou_mat = box_iou(torch.from_numpy(boxes_xyxy), torch.from_numpy(boxes_xyxy))
            iou_mat.fill_diagonal_(0.0)
            isolated_list = (iou_mat.amax(dim=1) < 0.20).tolist()
        else:
            isolated_list = [True] * len(boxes_xyxy)

        raw_detections = [
            ([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(conf), "person")
            for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs)
        ]

        tracks = self.tracker.update_tracks(raw_detections, embeds=feats, others=isolated_list)
        self._suppress_duplicate_tracks(tracks)

        # Retire unmatched local tracks whose Global ID has already moved to another camera
        if hasattr(self.global_manager, "get_identity_current_camera"):
            for tr in tracks:
                lid = int(tr.track_id)
                if not tr.is_deleted() and tr.time_since_update > 0 and lid in self.local_to_global_map:
                    gid = self.local_to_global_map[lid]
                    try:
                        owner_cam = self.global_manager.get_identity_current_camera(gid)
                        if owner_cam is not None and owner_cam != self.cam_id:
                            del self.local_to_global_map[lid]
                            tr.state = TrackState.Deleted
                    except Exception:
                        pass

        alive_local_ids = {int(t.track_id) for t in tracks if not t.is_deleted()}
        for lid in list(self.local_to_global_map.keys()):
            if lid not in alive_local_ids:
                del self.local_to_global_map[lid]

        active_global_ids = {
            self.local_to_global_map[int(t.track_id)]
            for t in tracks
            if not t.is_deleted() and int(t.track_id) in self.local_to_global_map and t.time_since_update == 0
        }

        render_candidates = [t for t in tracks if t.is_confirmed() and not t.is_deleted()]
        render_candidates.sort(
            key=lambda t: (
                0 if t.time_since_update == 0 else 1,
                0 if int(t.track_id) in self.local_to_global_map else 1,
                -t.hits,
                int(t.track_id),
            )
        )

        if len(render_candidates) > 1:
            cand_boxes = torch.tensor([list(t.to_ltrb()) for t in render_candidates], dtype=torch.float32)
            c_iou = box_iou(cand_boxes, cand_boxes)
            c_iou.fill_diagonal_(0.0)
            cand_isolated = (c_iou.amax(dim=1) < 0.18).tolist()
        else:
            cand_isolated = [True] * len(render_candidates)

        do_feature_update = (self.frame_counter % self.feature_update_interval == 0)
        rendered_global_ids: set[int] = set()
        current_annotations: list[AnnotationSpec] = []

        for idx_c, track in enumerate(render_candidates):
            box = [float(v) for v in track.to_ltrb()]
            if track.time_since_update > 0:
                if (
                    track.time_since_update > self.max_coast_frames
                    or track.hits < 3
                    or self._is_exiting_border(box)
                ):
                    continue

            local_id = int(track.track_id)
            ema_feature = track.get_feature()
            is_isolated = cand_isolated[idx_c]

            if local_id not in self.local_to_global_map:
                global_id = self.global_manager.get_or_create_global_id(
                    new_feature_vector=ema_feature,
                    bbox=box,
                    cam_id=self.cam_id,
                    current_time=current_time,
                    active_global_ids=active_global_ids | rendered_global_ids,
                )
                # Retire any older unmatched local track in this camera that previously held this Global ID
                for other_t in tracks:
                    other_lid = int(other_t.track_id)
                    if other_lid != local_id and self.local_to_global_map.get(other_lid) == global_id:
                        del self.local_to_global_map[other_lid]
                        if other_t.time_since_update > 0:
                            other_t.state = TrackState.Deleted

                self.local_to_global_map[local_id] = global_id
                active_global_ids.add(global_id)
            else:
                global_id = self.local_to_global_map[local_id]
                if track.time_since_update == 0 and (do_feature_update or track.hits <= 12):
                    feat_for_global = ema_feature if is_isolated else None
                    resolved_id = self.global_manager.update_existing_identity(
                        global_id=global_id,
                        new_feature_vector=feat_for_global,
                        bbox=box,
                        cam_id=self.cam_id,
                        current_time=current_time,
                        active_global_ids=active_global_ids | rendered_global_ids,
                    )
                    if resolved_id is None:
                        active_global_ids.discard(global_id)
                        global_id = self.global_manager.get_or_create_global_id(
                            new_feature_vector=ema_feature,
                            bbox=box,
                            cam_id=self.cam_id,
                            current_time=current_time,
                            active_global_ids=active_global_ids | rendered_global_ids,
                        )
                        self.local_to_global_map[local_id] = global_id
                        active_global_ids.add(global_id)
                    elif resolved_id is not None and resolved_id != global_id:
                        active_global_ids.discard(global_id)
                        global_id = resolved_id
                        for other_t in tracks:
                            other_lid = int(other_t.track_id)
                            if other_lid != local_id and self.local_to_global_map.get(other_lid) == global_id:
                                del self.local_to_global_map[other_lid]
                                if other_t.time_since_update > 0:
                                    other_t.state = TrackState.Deleted
                        self.local_to_global_map[local_id] = global_id
                        active_global_ids.add(global_id)

            if global_id in rendered_global_ids:
                continue
            rendered_global_ids.add(global_id)

            context = TrackContext(
                global_id=global_id,
                local_id=local_id,
                cam_id=self.cam_id,
                bbox=box,
                current_time=current_time,
                confidence=float(track.det_conf or 1.0),
            )
            spec = self.behavior_pipeline.evaluate(
                context=context,
                global_manager=self.global_manager,
                force_sync=do_feature_update,
            )
            current_annotations.append(spec)
            draw_annotation_spec(frame, spec, self.config)

        self._last_annotations = current_annotations
        draw_camera_hud(frame, self.cam_id, self.fps, len(current_annotations))
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        engine = self._get_or_init_engine()
        boxes_xyxy, confs, feats = engine.infer_batch([frame])[0]
        return self.process_detections(frame, boxes_xyxy, confs, feats)

    def cleanup(self) -> None:
        print(f"[Worker-{self.cam_id}] Recursos libertados.")
