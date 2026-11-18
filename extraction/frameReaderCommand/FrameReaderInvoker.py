# extraction/frameReaderCommand/FrameReaderInvoker.py
import logging
import threading
from typing import Optional
import numpy as np

from extraction.frameReaderCommand.IFrameCommand import IFrameCommand


class FrameReaderInvoker(threading.Thread):
    """Background I/O thread that executes an `IFrameCommand` and keeps the freshest frame per camera."""

    def __init__(self, cam_key: str, command: IFrameCommand, stop_event: threading.Event, name: str = "FrameReaderInvoker"):
        super().__init__(daemon=True, name=name)
        self.cam_key = cam_key
        self.command = command
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self._latest_frames: dict[str, np.ndarray] = {}
        self._cached_frames: dict[str, np.ndarray] = {}
        self._seq_id = 0

    def _store_frames(self, batch: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._latest_frames.update(batch)
            self._cached_frames.update(batch)
            self._seq_id += 1

    def _step_capture(self, has_batch_exec: bool) -> None:
        if has_batch_exec:
            batch = self.command.execute_batch()
            if batch:
                self._store_frames(batch)
            return
        res = self.command.execute()
        if res is not None:
            self._store_frames({res[0]: res[1]})

    def run(self) -> None:
        logging.info(f"[{self.name}] Iniciando loop de captura para '{self.cam_key}'...")
        has_batch_exec = hasattr(self.command, "execute_batch")
        while not self.stop_event.is_set():
            self._step_capture(has_batch_exec)
        logging.info(f"[{self.name}] Thread encerrada. Executando limpeza...")
        self.command.cleanup()

    def get_latest(self) -> Optional[dict[str, np.ndarray]]:
        with self._lock:
            if not self._latest_frames:
                return None
            result = {cid: frm.copy() for cid, frm in self._latest_frames.items()}
            self._latest_frames.clear()
            return result

    def get_batch_state(self) -> tuple[Optional[dict[str, np.ndarray]], set[str]]:
        with self._lock:
            if not self._cached_frames:
                return None, set()
            new_cams = set(self._latest_frames.keys())
            self._latest_frames.clear()
            return {cid: frm.copy() for cid, frm in self._cached_frames.items()}, new_cams

    def get_latest_with_fallback(self) -> tuple[Optional[dict[str, np.ndarray]], bool]:
        with self._lock:
            is_new = bool(self._latest_frames)
            self._latest_frames.clear()
            if not self._cached_frames:
                return None, False
            return {cid: frm.copy() for cid, frm in self._cached_frames.items()}, is_new
