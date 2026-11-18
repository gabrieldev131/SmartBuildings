# extraction/frameReaderCommand/RTSPCommandFactory.py
from typing import Optional
from interfaces.ICommandFactory import ICommandFactory
from interfaces.IFrameCommand import IFrameCommand


class RTSPCommandFactory(ICommandFactory):
    """Picklable factory for creating `ReadRTSPCommand` instances across processes."""

    def __init__(self, source: str, width: int, height: int, reconnect_delay: int = 5, cam_id: Optional[str] = None, loop_video: bool = True, pace_fps: bool = False):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.cam_id = cam_id or source
        self.loop_video = loop_video
        self.pace_fps = pace_fps

    def __call__(self) -> IFrameCommand:
        from extraction.frameReaderCommand.ReadRTSPCommand import ReadRTSPCommand
        return ReadRTSPCommand(
            source=self.source, width=self.width, height=self.height,
            reconnect_delay=self.reconnect_delay, cam_id=self.cam_id,
            loop_video=self.loop_video, pace_fps=self.pace_fps,
        )
