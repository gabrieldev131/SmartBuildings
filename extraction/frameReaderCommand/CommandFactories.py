# extraction/frameReaderCommand/CommandFactories.py
"""
Picklable Command Factories (Abstract Factory Pattern) for IFrameCommand instantiation.
Enables clean dependency injection and serialization across process boundaries when
using multiprocessing with 'spawn'.
"""
from typing import Optional
from core.ports import ICommandFactory, IFrameCommand


class RTSPCommandFactory(ICommandFactory):
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
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.cam_id = cam_id or source
        self.loop_video = loop_video
        self.pace_fps = pace_fps

    def __call__(self) -> IFrameCommand:
        from extraction.frameReaderCommand.ReadRTSPCommand import ReadRTSPCommand
        return ReadRTSPCommand(
            source=self.source,
            width=self.width,
            height=self.height,
            reconnect_delay=self.reconnect_delay,
            cam_id=self.cam_id,
            loop_video=self.loop_video,
            pace_fps=self.pace_fps,
        )


class KafkaCommandFactory(ICommandFactory):
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        width: int,
        height: int,
        target_camera_id: str = "",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.width = width
        self.height = height
        self.target_camera_id = target_camera_id

    def __call__(self) -> IFrameCommand:
        from extraction.frameReaderCommand.ReadKafkaCommand import ReadKafkaCommand
        return ReadKafkaCommand(
            bootstrap_servers=self.bootstrap_servers,
            topic=self.topic,
            group_id=self.group_id,
            width=self.width,
            height=self.height,
            target_camera_id=self.target_camera_id,
        )
