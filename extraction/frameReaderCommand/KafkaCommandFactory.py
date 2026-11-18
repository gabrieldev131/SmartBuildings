# extraction/frameReaderCommand/KafkaCommandFactory.py
from interfaces.ICommandFactory import ICommandFactory
from interfaces.IFrameCommand import IFrameCommand


class KafkaCommandFactory(ICommandFactory):
    """Picklable factory for creating `ReadKafkaCommand` instances across processes."""

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str, width: int, height: int, target_camera_id: str = ""):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.width = width
        self.height = height
        self.target_camera_id = target_camera_id

    def __call__(self) -> IFrameCommand:
        from extraction.frameReaderCommand.ReadKafkaCommand import ReadKafkaCommand
        return ReadKafkaCommand(
            bootstrap_servers=self.bootstrap_servers, topic=self.topic,
            group_id=self.group_id, width=self.width, height=self.height,
            target_camera_id=self.target_camera_id,
        )
