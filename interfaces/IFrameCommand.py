# interfaces/IFrameCommand.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class IFrameCommand(ABC):
    """Command Pattern interface for reading video frames from RTSP, local files, or Kafka (ISP)."""

    @abstractmethod
    def execute(self) -> Optional[Tuple[str, Any]]:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass
