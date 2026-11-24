# vision/StabilizedTrack.py
"""
StabilizedTrack: Extension of `deep_sort_realtime.deep_sort.track.Track`
providing velocity damping on occlusion, EMA box smoothing, outlier-filtered appearance,
and full YOLO + DeepSort telemetry extraction.
"""
import numpy as np
from deep_sort_realtime.deep_sort.track import Track, TrackState


class StabilizedTrack(Track):
    """Subclass of `deep-sort-realtime` `Track` passed via `override_track_class`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        feat = self.features[-1] if self.features else np.zeros(576, dtype=np.float32)
        self.ema_feat = self._normalize(feat)
        self.smooth_ltrb = list(super().to_ltrb(orig=True))
        self.max_conf = float(self.det_conf or 0.0)
        self.last_appearance_sim = 1.0
        self.is_isolated_track = True
        self.class_id = 0

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-6 else arr

    def predict(self, kf) -> None:
        self.mean[6] = 0.0
        if 2 < self.time_since_update <= 6:
            self.mean[4:6] *= 0.65
            self.mean[7] = 0.0
        elif self.time_since_update > 6:
            self.mean[4:8] = 0.0

        super().predict(kf)
        self.mean[2] = float(np.clip(self.mean[2], 0.14, 2.85))
        self.mean[3] = max(16.0, float(self.mean[3]))
        self._update_coasting_box()

    def _update_coasting_box(self) -> None:
        if self.time_since_update not in (1, 2, 3):
            return
        damp = 0.45 if self.time_since_update == 1 else (0.25 if self.time_since_update == 2 else 0.10)
        vx, vy = float(self.mean[4]) * damp, float(self.mean[5]) * damp
        x1, y1, x2, y2 = self.smooth_ltrb
        self.smooth_ltrb = [x1 + vx, y1 + vy, x2 + vx, y2 + vy]

    def mark_missed(self) -> None:
        if self.state == TrackState.Tentative:
            if self.time_since_update > 2:
                self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def update(self, kf, detection) -> None:
        prev_feat = self.ema_feat
        super().update(kf, detection)
        self.mean[6] = 0.0
        self.max_conf = max(self.max_conf, float(detection.confidence or 0.0))
        if self.state == TrackState.Confirmed and self.max_conf < 0.48:
            self.state = TrackState.Tentative

        self._smooth_bounding_box()
        self._update_appearance(detection, prev_feat)

    def _smooth_bounding_box(self) -> None:
        det_box = np.asarray(super().to_ltrb(orig=True), dtype=np.float64)
        kalman_box = np.asarray(super().to_ltrb(orig=False), dtype=np.float64)
        target = 0.72 * det_box + 0.28 * kalman_box
        prev = np.asarray(self.smooth_ltrb, dtype=np.float64)
        h = max(16.0, float(target[3] - target[1]))
        shift = float(np.hypot(0.5 * (target[0] + target[2] - prev[0] - prev[2]), 0.5 * (target[1] + target[3] - prev[1] - prev[3])))
        alpha = min(0.92, 0.65 + 0.35 * min(1.0, shift / (0.12 * h)))
        self.smooth_ltrb = [float(x) for x in (alpha * target + (1.0 - alpha) * prev)]

    def _update_appearance(self, detection, prev_feat: np.ndarray) -> None:
        meta = detection.others if isinstance(detection.others, dict) else {"isolated": bool(detection.others) if detection.others is not None else True}
        self.is_isolated_track = bool(meta.get("isolated", True))
        self.class_id = int(meta.get("class_id", self.class_id))
        if not self.is_isolated_track or detection.feature is None:
            self.features[-1] = prev_feat
            return
        unit_f = self._normalize(detection.feature)
        self.last_appearance_sim = float(np.clip(np.dot(prev_feat, unit_f), 0.0, 1.0))
        if self.hits <= 4 or self.last_appearance_sim >= 0.50:
            w = 0.55 if self.hits <= 4 else 0.82
            self.ema_feat = self._normalize(w * prev_feat + (1.0 - w) * unit_f)
        self.features[-1] = self.ema_feat

    def to_ltrb(self, orig: bool = False, orig_strict: bool = False) -> np.ndarray:
        return np.asarray(self.smooth_ltrb, dtype=np.float32)

    def get_feature(self) -> np.ndarray:
        return self.ema_feat

    def extract_yolo_metrics(self, frame_w: int, frame_h: int, is_near_border: bool, model_path: str = "") -> dict:
        x1, y1, x2, y2 = [float(v) for v in self.smooth_ltrb]
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        cx, cy, area = (x1 + x2) * 0.5, (y1 + y2) * 0.5, w * h
        conf = float(self.det_conf or 0.0)
        raw_ltwh = [round(float(v), 2) for v in (self.original_ltwh if self.original_ltwh is not None else [x1, y1, w, h])]
        return {
            "object_class": str(self.det_class or "person"), "class_id": int(self.class_id),
            "confidence": round(conf, 4), "confidence_pct": round(conf * 100.0, 2),
            "max_confidence": round(float(self.max_conf), 4), "max_confidence_pct": round(float(self.max_conf) * 100.0, 2),
            "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "bbox_ltwh": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)], "raw_bbox_ltwh": raw_ltwh,
            "center_xy": [round(cx, 2), round(cy, 2)],
            "center_norm": [round(cx / max(1, frame_w), 4), round(cy / max(1, frame_h), 4)],
            "width": round(w, 2), "height": round(h, 2), "area_px": round(area, 2),
            "area_pct": round((area / max(1.0, float(frame_w * frame_h))) * 100.0, 2),
            "aspect_ratio": round(w / max(1.0, h), 4), "is_near_border": bool(is_near_border), "model_path": model_path,
        }

    def extract_deepsort_metrics(self, global_id: int, fps: float) -> dict:
        vx, vy = float(self.mean[4]), float(self.mean[5])
        speed = float(np.hypot(vx, vy))
        state_str = "Confirmed" if self.is_confirmed() else ("Tentative" if self.is_tentative() else "Deleted")
        return {
            "local_track_id": int(self.track_id), "global_id": int(global_id), "track_state": state_str,
            "is_confirmed": bool(self.is_confirmed()), "is_tentative": bool(self.is_tentative()),
            "hits": int(self.hits), "age": int(self.age), "time_since_update": int(self.time_since_update),
            "is_coasting": bool(self.time_since_update > 0), "is_isolated": bool(self.is_isolated_track),
            "velocity_vx": round(vx, 4), "velocity_vy": round(vy, 4),
            "speed_px_per_frame": round(speed, 4), "speed_px_per_sec": round(speed * max(0.0, fps), 2),
            "movement_direction_deg": round(float(np.degrees(np.arctan2(vy, vx))), 2),
            "kalman_mean_8d": [round(float(x), 4) for x in self.mean[:8]],
            "kalman_bbox_ltrb": [round(float(x), 2) for x in super().to_ltrb(orig=False)],
            "smoothed_bbox_ltrb": [round(float(x), 2) for x in self.smooth_ltrb],
            "position_uncertainty": round(float(np.trace(self.covariance[:4, :4])), 4),
            "velocity_uncertainty": round(float(np.trace(self.covariance[4:8, 4:8])), 4),
            "appearance_similarity_pct": round(float(self.last_appearance_sim) * 100.0, 2),
            "embedding_dim": int(len(self.ema_feat)), "embedding_norm": round(float(np.linalg.norm(self.ema_feat)), 4), "gallery_size": int(len(self.features)),
        }
