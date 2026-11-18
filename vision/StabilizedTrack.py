# vision/StabilizedTrack.py
"""
StabilizedTrack: Extension of `deep_sort_realtime.deep_sort.track.Track`
providing velocity damping on occlusion, EMA box smoothing, and outlier-filtered appearance.
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
        is_isolated = bool(detection.others) if detection.others is not None else True
        if not is_isolated or detection.feature is None:
            self.features[-1] = prev_feat
            return
        unit_f = self._normalize(detection.feature)
        if self.hits <= 4 or float(np.dot(prev_feat, unit_f)) >= 0.50:
            w = 0.55 if self.hits <= 4 else 0.82
            self.ema_feat = self._normalize(w * prev_feat + (1.0 - w) * unit_f)
        self.features[-1] = self.ema_feat

    def to_ltrb(self, orig: bool = False, orig_strict: bool = False) -> np.ndarray:
        return np.asarray(self.smooth_ltrb, dtype=np.float32)

    def get_feature(self) -> np.ndarray:
        return self.ema_feat
