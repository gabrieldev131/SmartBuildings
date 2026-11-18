# extraction/frameReaderCommand/ReadRTSPCommand.py
import logging
import os
import time
from typing import Any, Optional, Tuple
import cv2

from extraction.frameReaderCommand.IFrameCommand import IFrameCommand


class ReadRTSPCommand(IFrameCommand):
    """Concrete Command for reading frames from RTSP streams or local video files (SRP & LSP)."""

    def __init__(self, source: str, width: int, height: int, reconnect_delay: int = 5, cam_id: Optional[str] = None, loop_video: bool = True, pace_fps: bool = False):
        self.source = source
        self.cam_id = cam_id or source
        self.width, self.height = width, height
        self.reconnect_delay = reconnect_delay
        self.loop_video, self.pace_fps = loop_video, pace_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_network = str(source).lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://"))
        self._frame_interval, self._last_read_ts = 0.0, 0.0

    def _connect(self) -> None:
        logging.info(f"Tentando conectar a '{self.source}' (cam_id='{self.cam_id}')...")
        opts = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|threads;2" if self._is_network else "threads;2"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            logging.error(f"Erro ao abrir fonte de vídeo: {self.source}")
            self.cap = None
            return
        if not self._is_network and self.pace_fps:
            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._frame_interval = (1.0 / float(src_fps)) if (src_fps and src_fps > 1.0) else 0.0

    def _pace_if_needed(self) -> None:
        if self._frame_interval <= 0.0:
            return
        wait = self._frame_interval - (time.perf_counter() - self._last_read_ts)
        if wait > 0.0005:
            time.sleep(wait)
        self._last_read_ts = time.perf_counter()

    def _read_raw_frame(self):
        ok, frame = self.cap.read()
        if (not ok or frame is None) and not self._is_network and self.loop_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def execute(self) -> Optional[Tuple[str, Any]]:
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            if self.cap is None:
                time.sleep(self.reconnect_delay)
                return None
        self._pace_if_needed()
        ok, frame = self._read_raw_frame()
        if not ok or frame is None:
            logging.warning(f"Falha na leitura de frame em {self.source}. Tentando reconectar...")
            self.cleanup()
            return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return (self.cam_id, frame)

    def cleanup(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
