# ui/display.py
import math
import time
import cv2
import numpy as np


def _draw_labeled_box(frame: np.ndarray, box: list[float], label: str, color: tuple[int, int, int], font_scale: float) -> None:
    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, min(w_img - 1, int(box[0]))), max(0, min(h_img - 1, int(box[1])))
    x2, y2 = max(0, min(w_img - 1, int(box[2]))), max(0, min(h_img - 1, int(box[3])))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    top_y = max(th + 6, y1)
    cv2.rectangle(frame, (x1, top_y - th - 6), (x1 + tw + 4, top_y), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 2, top_y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def draw_annotation_spec(frame: np.ndarray, spec, config) -> None:
    """Draw bounding box and dynamic badge from a `BehaviorPipeline` `AnnotationSpec`."""
    _draw_labeled_box(frame, spec.bbox, spec.label, spec.color, config.FONT_SCALE)


def draw_person_annotation(frame: np.ndarray, box: list[float], global_id: int, is_stopped: bool, elapsed: float, config) -> None:
    """Legacy helper drawing bounding box and stopped state badge."""
    color = config.COLOR_STOPPED if is_stopped else config.COLOR_MOVING
    label = f"ID:{global_id}" + (f" PARADO {int(elapsed)}s" if is_stopped else "")
    _draw_labeled_box(frame, box, label, color, config.FONT_SCALE)


def draw_camera_hud(frame: np.ndarray, cam_id: str, fps: float, active_tracks: int = 0) -> None:
    """Render FPS, active person count, and camera identifier overlay."""
    short_name = (cam_id.split("/")[-1] if "/" in cam_id else cam_id)[:22]
    cv2.rectangle(frame, (5, 5), (235, 58), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f} | Pessoas: {active_tracks}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Cam: {short_name}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


class MultiCameraDisplay:
    """Manages OpenCV GUI presentation for single or tiled multi-camera views (SRP)."""

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, "DISPLAY_ENABLED", True)
        self.grid_view = getattr(config, "DISPLAY_GRID_VIEW", False)
        max_fps = getattr(config, "DISPLAY_MAX_FPS", 60)
        self.min_render_interval = (1.0 / max_fps) if max_fps > 0 else 0.0
        self._last_render_time = 0.0
        self._windows_created: set[str] = set()
        self._grid_window_name = "SmartBuildings - Multi-Camera Dashboard"
        self._latest_frames_by_cam: dict[str, np.ndarray] = {}

    def _render_grid_window(self, active_frames: dict[str, np.ndarray]) -> None:
        for win in [w for w in list(self._windows_created) if w != self._grid_window_name]:
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass
            self._windows_created.discard(win)
        canvas = self._build_grid(active_frames)
        if self._grid_window_name not in self._windows_created:
            cv2.namedWindow(self._grid_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._grid_window_name, min(1920, canvas.shape[1]), min(1080, canvas.shape[0]))
            self._windows_created.add(self._grid_window_name)
        cv2.imshow(self._grid_window_name, canvas)

    def _render_individual_windows(self, active_frames: dict[str, np.ndarray]) -> None:
        for cam_id, frame in active_frames.items():
            win = f"SmartBuilds - {cam_id}"
            if win not in self._windows_created:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, self.config.PROCESSING_WIDTH, self.config.PROCESSING_HEIGHT)
                self._windows_created.add(win)
            cv2.imshow(win, frame)

    def render(self, frames_by_cam: dict[str, np.ndarray]) -> bool:
        if not self.enabled or not frames_by_cam:
            return True
        self._latest_frames_by_cam.update(frames_by_cam)
        now = time.perf_counter()
        if (now - self._last_render_time) < self.min_render_interval:
            return True
        self._last_render_time = now
        if self.grid_view and len(self._latest_frames_by_cam) > 1:
            self._render_grid_window(self._latest_frames_by_cam)
        else:
            self._render_individual_windows(self._latest_frames_by_cam)
        return self.poll_key()

    def poll_key(self) -> bool:
        if not self.enabled or not self._windows_created:
            return True
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    @staticmethod
    def _build_grid(frames_by_cam: dict[str, np.ndarray]) -> np.ndarray:
        cam_keys = sorted(frames_by_cam.keys())
        n = len(cam_keys)
        cols = math.ceil(math.sqrt(n)) if n <= 4 else 4
        rows = math.ceil(n / cols)
        h, w = frames_by_cam[cam_keys[0]].shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for idx, k in enumerate(cam_keys):
            r, c, frm = idx // cols, idx % cols, frames_by_cam[k]
            if frm.shape[0] != h or frm.shape[1] != w:
                frm = cv2.resize(frm, (w, h), interpolation=cv2.INTER_LINEAR)
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = frm
        return grid

    def cleanup(self) -> None:
        for win in list(self._windows_created):
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass
        self._windows_created.clear()
