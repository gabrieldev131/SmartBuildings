# extraction/frameReaderCommand/FrameReaderInvoker.py
import logging
import threading
from typing import Optional
import numpy as np

from extraction.frameReaderCommand.IFrameCommand import IFrameCommand


class FrameReaderInvoker(threading.Thread):
    """
    Command Invoker (Command Pattern) that continuously executes an IFrameCommand
    on a dedicated background I/O thread and maintains only the freshest frame per camera.
    Prevents queue buildup and latency drift when capture and inference run at different rates.
    """

    def __init__(
        self,
        cam_key: str,
        command: IFrameCommand,
        stop_event: threading.Event,
        name: str = "FrameReaderInvoker",
    ):
        super().__init__(daemon=True, name=name)
        self.cam_key = cam_key
        self.command = command
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self._latest_frames: dict[str, np.ndarray] = {}
        self._cached_frames: dict[str, np.ndarray] = {}
        self._seq_id = 0

    def run(self) -> None:
        logging.info(f"[{self.name}] Iniciando loop de captura para '{self.cam_key}'...")
        has_batch_exec = hasattr(self.command, "execute_batch")

        while not self.stop_event.is_set():
            if has_batch_exec:
                batch = self.command.execute_batch()
                if batch:
                    with self._lock:
                        for cam_id, frame in batch.items():
                            self._latest_frames[cam_id] = frame
                            self._cached_frames[cam_id] = frame
                        self._seq_id += 1
            else:
                result = self.command.execute()
                if result is not None:
                    cam_id, frame = result
                    with self._lock:
                        self._latest_frames[cam_id] = frame
                        self._cached_frames[cam_id] = frame
                        self._seq_id += 1

        logging.info(f"[{self.name}] Thread encerrada. Executando limpeza...")
        self.command.cleanup()

    def get_latest(self) -> Optional[dict[str, np.ndarray]]:
        """
        Consume and return the dictionary of newly arrived frames since the last call.
        Returns None if no new frame has been decoded yet.
        """
        with self._lock:
            if not self._latest_frames:
                return None
            result = {cid: frm.copy() for cid, frm in self._latest_frames.items()}
            self._latest_frames.clear()
            return result

    def get_latest_with_fallback(self) -> tuple[Optional[dict[str, np.ndarray]], bool]:
        """
        Return (frames_dict, is_new). If a new frame arrived, consumes it and returns (frames, True).
        Otherwise returns the most recently cached frame (frames, False) so multi-camera batching
        never stalls waiting on a slower camera stream.
        """
        with self._lock:
            if self._latest_frames:
                self._latest_frames.clear()
                return {cid: frm.copy() for cid, frm in self._cached_frames.items()}, True
            if self._cached_frames:
                return {cid: frm.copy() for cid, frm in self._cached_frames.items()}, False
            return None, False
