# vision/cameraWorker.py
"""
CameraWorker: Per-camera tracking, global identity resolution, behavioral state evaluation,
and visual annotation.
Supports both:
1) Batched Pipeline Mode via `process_detections(...)` (where detection & Re-ID are
   executed across all cameras simultaneously by BatchInferenceEngine).
2) Standalone Mode via `process_frame(...)` (for single-camera or multiprocess callers).
"""
import time
from typing import Optional
import numpy as np

from core.StoppedStateTracker import StoppedStateTracker
from ui.display import draw_person_annotation, draw_camera_hud
from vision.VectorizedDeepSort import VectorizedDeepSort


class CameraWorker:
    """
    Worker responsible for associating detections into stabilized local tracks (VectorizedDeepSort),
    resolving cross-camera identities (GlobalIdentityManager), evaluating stopped behavior
    (StoppedStateTracker), and annotating output frames.
    """

    def __init__(self, cam_id: str, config, global_manager, inference_engine=None):
        self.cam_id = cam_id
        self.config = config
        self.global_manager = global_manager
        self._inference_engine = inference_engine

        self.local_to_global_map: dict[int, int] = {}
        self._prev_stopped_map: dict[int, bool] = {}
        self.state_tracker = StoppedStateTracker(self.config)

        self.tracker = VectorizedDeepSort(
            max_age=getattr(config, "DEEPSORT_MAX_AGE", 30),
            n_init=getattr(config, "DEEPSORT_N_INIT", 3),
            max_iou_distance=getattr(config, "DEEPSORT_MAX_IOU_DISTANCE", 0.55),
            max_cosine_distance=getattr(config, "DEEPSORT_MAX_COSINE_DISTANCE", 0.26),
            nn_budget=getattr(config, "DEEPSORT_NN_BUDGET", 30),
            conf_high=getattr(config, "YOLO_CONF_HIGH", 0.45),
            conf_low=getattr(config, "YOLO_CONF_LOW", 0.28),
            init_conf=getattr(config, "YOLO_INIT_CONF", 0.52),
            box_smooth_alpha=getattr(config, "DEEPSORT_BOX_SMOOTH_ALPHA", 0.65),
        )

        self.max_coast_frames = getattr(config, "DEEPSORT_MAX_COAST_FRAMES", 1)
        self.feature_update_interval = getattr(config, "FEATURE_UPDATE_INTERVAL", 3)
        self.frame_w = getattr(config, "PROCESSING_WIDTH", 640)
        self.frame_h = getattr(config, "PROCESSING_HEIGHT", 480)
        self.frame_counter = 0

        self.prev_time = time.perf_counter()
        self.fps = 0.0

        print(
            f"[Worker-{self.cam_id}] Inicializado. "
            f"GlobalIdentityManager compartilhado id={id(self.global_manager)}"
        )

    def _get_or_init_engine(self):
        if self._inference_engine is None:
            from vision.BatchInferenceEngine import BatchInferenceEngine
            self._inference_engine = BatchInferenceEngine(self.config)
        return self._inference_engine

    def _is_exiting_border(self, box: list[float]) -> bool:
        """Return True if box is clipping the image border (prevents coasting off-screen)."""
        x1, y1, x2, y2 = box
        margin = 6.0
        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= (self.frame_w - margin)
            or y2 >= (self.frame_h - margin)
        )

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
        """
        Process pre-extracted detections and Re-ID embeddings for this camera.
        """
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

        if is_new_frame:
            self.frame_counter += 1
            tracks = self.tracker.update(boxes_xyxy, confs, feats)
        else:
            tracks = self.tracker.tracks

        # Retain local_to_global_map only for non-deleted tracks
        alive_local_ids = {track.track_id for track in tracks if not track.is_deleted()}
        for lid in list(self.local_to_global_map.keys()):
            if lid not in alive_local_ids:
                del self.local_to_global_map[lid]

        # Only block Global IDs that are currently visible or missed for <= 4 frames in this camera,
        # so if a track was lost and re-detected as a new local track, it can reclaim its own Global ID
        active_global_ids = {
            self.local_to_global_map[t.track_id]
            for t in tracks
            if not t.is_deleted()
            and t.track_id in self.local_to_global_map
            and t.time_since_update <= 4
        }

        do_feature_update = is_new_frame and (self.frame_counter % self.feature_update_interval == 0)

        active_count = 0
        rendered_global_ids: set[int] = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            box = track.to_ltrb()
            # Only allow 1-frame coasting on mature interior tracks; never coast at frame borders
            if track.time_since_update > 0:
                if (
                    track.time_since_update > self.max_coast_frames
                    or track.hits < 6
                    or self._is_exiting_border(box)
                ):
                    continue

            local_id = track.track_id
            ema_feature = track.get_feature()

            if local_id not in self.local_to_global_map:
                global_id = self.global_manager.get_or_create_global_id(
                    new_feature_vector=ema_feature,
                    bbox=box,
                    cam_id=self.cam_id,
                    current_time=current_time,
                    active_global_ids=active_global_ids | rendered_global_ids,
                )
                self.local_to_global_map[local_id] = global_id
                active_global_ids.add(global_id)
            else:
                global_id = self.local_to_global_map[local_id]
                if do_feature_update and track.time_since_update == 0:
                    self.global_manager.update_existing_identity(
                        global_id=global_id,
                        new_feature_vector=ema_feature,
                        bbox=box,
                        cam_id=self.cam_id,
                        current_time=current_time,
                    )

            # Guarantee two boxes in the same camera frame never render with the same Global ID
            if global_id in rendered_global_ids:
                continue
            rendered_global_ids.add(global_id)
            active_count += 1

            is_stopped, elapsed = self.state_tracker.update_and_evaluate(global_id, box, current_time)
            was_stopped = self._prev_stopped_map.get(global_id, False)
            if is_stopped or was_stopped or do_feature_update:
                if hasattr(self.global_manager, "record_stopped_state"):
                    self.global_manager.record_stopped_state(
                        global_id=global_id,
                        cam_id=self.cam_id,
                        is_stopped=is_stopped,
                        elapsed=elapsed,
                        current_time=current_time,
                    )
            self._prev_stopped_map[global_id] = is_stopped

            draw_person_annotation(frame, box, global_id, is_stopped, elapsed, self.config)

        draw_camera_hud(frame, self.cam_id, self.fps, active_count)
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Standalone single-camera entrypoint: runs detection + Re-ID extraction and tracks.
        """
        engine = self._get_or_init_engine()
        batch_out = engine.infer_batch([frame])
        boxes_xyxy, confs, feats = batch_out[0]
        return self.process_detections(frame, boxes_xyxy, confs, feats)

    @staticmethod
    def normalize_feature(feature_vector: np.ndarray) -> np.ndarray:
        """Ensure L2 norm = 1.0 for cosine similarity calculations."""
        norm = float(np.linalg.norm(feature_vector))
        if norm <= 1e-6:
            return feature_vector
        return (feature_vector / norm).astype(np.float32)

    def cleanup(self) -> None:
        print(f"[Worker-{self.cam_id}] Recursos libertados.")
