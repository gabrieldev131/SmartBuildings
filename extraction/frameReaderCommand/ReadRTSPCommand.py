# extraction/frameReaderCommand/ReadRTSPCommand.py
import logging
import os
import time
from typing import Any, Optional, Tuple
import cv2

from extraction.frameReaderCommand.IFrameCommand import IFrameCommand


class ReadRTSPCommand(IFrameCommand):
    """
    Concrete Command (Command Pattern) responsible for reading frames from an RTSP network stream
    or local video file with automatic reconnection/looping and fast resolution normalization.
    """

    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        reconnect_delay: int = 5,
        cam_id: Optional[str] = None,
        loop_video: bool = True,
        pace_fps: bool = False,
    ):
        self.source = source
        self.cam_id = cam_id or source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.loop_video = loop_video
        self.pace_fps = pace_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_network = str(source).lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://"))
        self._frame_interval = 0.0
        self._last_read_ts = 0.0

    def _connect(self) -> None:
        logging.info(f"Tentando conectar a '{self.source}' (cam_id='{self.cam_id}')...")

        if self._is_network:
            # Low-latency TCP transport with 2 decoder threads for RTSP streams
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|threads;2"
            )
        else:
            # Local high-resolution files (e.g., 4K 60fps MP4) benefit from 2-thread FFmpeg decoding
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;2"

        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            logging.error(f"Erro ao abrir fonte de vídeo: {self.source}")
            self.cap = None
            return

        if not self._is_network and self.pace_fps:
            src_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if src_fps and src_fps > 1.0:
                self._frame_interval = 1.0 / float(src_fps)

    def execute(self) -> Optional[Tuple[str, Any]]:
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            if self.cap is None:
                time.sleep(self.reconnect_delay)
                return None

        if self._frame_interval > 0.0:
            now = time.perf_counter()
            wait = self._frame_interval - (now - self._last_read_ts)
            if wait > 0.0005:
                time.sleep(wait)
            self._last_read_ts = time.perf_counter()

        success, frame = self.cap.read()
        if not success or frame is None:
            # For local video files, rewind immediately to frame 0 without tearing down the decoder
            if not self._is_network and self.loop_video:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()

            if not success or frame is None:
                logging.warning(f"Falha na leitura de frame em {self.source}. Tentando reconectar...")
                self.cleanup()
                return None

        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        return (self.cam_id, frame)

    def cleanup(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
