# ui/display.py
"""
Presentation Layer: OpenCV HUD Annotation & Multi-Camera Dashboard Presenter.
Decouples GUI window management from extraction and vision layers (Clean Architecture).
"""
import math
import time
import cv2
import numpy as np


def draw_person_annotation(frame: np.ndarray, box: list[float], global_id: int, is_stopped: bool, elapsed: float, config) -> None:
    """Draw bounding box and identity/state badge for a tracked person."""
    h_img, w_img = frame.shape[:2]
    x1 = max(0, min(w_img - 1, int(box[0])))
    y1 = max(0, min(h_img - 1, int(box[1])))
    x2 = max(0, min(w_img - 1, int(box[2])))
    y2 = max(0, min(h_img - 1, int(box[3])))

    color = config.COLOR_STOPPED if is_stopped else config.COLOR_MOVING
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{global_id}"
    if is_stopped:
        label += f" PARADO {int(elapsed)}s"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
    top_y = max(th + 6, y1)
    cv2.rectangle(frame, (x1, top_y - th - 6), (x1 + tw + 4, top_y), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, top_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_camera_hud(frame: np.ndarray, cam_id: str, fps: float, active_tracks: int = 0) -> None:
    """Render FPS, active person count, and camera identifier overlay."""
    short_name = cam_id.split("/")[-1] if "/" in cam_id else cam_id
    if len(short_name) > 22:
        short_name = short_name[:22]

    cv2.rectangle(frame, (5, 5), (235, 58), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Pessoas: {active_tracks}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Cam: {short_name}",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


class MultiCameraDisplay:
    """
    Manages OpenCV GUI presentation for 1 to 8+ cameras.
    Supports both a unified tiled Grid View (recommended for 2-8 cameras to eliminate
    X11 multi-window context-switch stalls) and individual per-camera windows.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, "DISPLAY_ENABLED", True)
        self.grid_view = getattr(config, "DISPLAY_GRID_VIEW", True)
        max_fps = getattr(config, "DISPLAY_MAX_FPS", 60)
        self.min_render_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_render_time = 0.0
        self._windows_created: set[str] = set()
        self._grid_window_name = "SmartBuildings - Multi-Camera Dashboard"
        self._latest_frames_by_cam: dict[str, np.ndarray] = {}

    def render(self, frames_by_cam: dict[str, np.ndarray]) -> bool:
        """
        Render the latest annotated frames.
        Retains the most recent annotated frame per camera so asynchronous streams
        (e.g., multiplexed Kafka cameras) never cause window flickering.
        Returns False if the user requested exit (pressed 'q' or ESC), True otherwise.
        """
        if not self.enabled or not frames_by_cam:
            return True

        self._latest_frames_by_cam.update(frames_by_cam)

        now = time.perf_counter()
        if (now - self._last_render_time) < self.min_render_interval:
            return True
        self._last_render_time = now

        active_frames = self._latest_frames_by_cam

        if self.grid_view and len(active_frames) > 1:
            # Close any single-camera windows if a second camera joined dynamically (e.g. via Kafka)
            for win in list(self._windows_created):
                if win != self._grid_window_name:
                    try:
                        cv2.destroyWindow(win)
                    except cv2.error:
                        pass
                    self._windows_created.discard(win)

            canvas = self._build_grid(active_frames)
            if self._grid_window_name not in self._windows_created:
                cv2.namedWindow(self._grid_window_name, cv2.WINDOW_NORMAL)
                h, w = canvas.shape[:2]
                cv2.resizeWindow(self._grid_window_name, min(1920, w), min(1080, h))
                self._windows_created.add(self._grid_window_name)
            cv2.imshow(self._grid_window_name, canvas)
        else:
            for cam_id, frame in active_frames.items():
                win_name = f"SmartBuilds - {cam_id}"
                if win_name not in self._windows_created:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, self.config.PROCESSING_WIDTH, self.config.PROCESSING_HEIGHT)
                    self._windows_created.add(win_name)
                cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            return False
        return True

    def poll_key(self) -> bool:
        """Pump GUI events when waiting for new frames. Returns False if user pressed exit."""
        if not self.enabled or not self._windows_created:
            return True
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            return False
        return True

    def _build_grid(self, frames_by_cam: dict[str, np.ndarray]) -> np.ndarray:
        cam_keys = sorted(frames_by_cam.keys())
        n = len(cam_keys)
        cols = math.ceil(math.sqrt(n)) if n <= 4 else 4
        rows = math.ceil(n / cols)

        sample = frames_by_cam[cam_keys[0]]
        h, w = sample.shape[:2]
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        for idx, k in enumerate(cam_keys):
            r = idx // cols
            c = idx % cols
            frm = frames_by_cam[k]
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
