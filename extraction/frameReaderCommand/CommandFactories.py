# extraction/frameReaderCommand/CommandFactories.py
"""
Factories picklable para IFrameCommand.

Por que isso existe: com multiprocessing usando "spawn" (obrigatório com
CUDA -- ver main.py), todo argumento passado para Process(..., args=...)
precisa ser serializável (pickle). Uma lambda não é picklable (o pickle
não consegue reconstruir uma função anônima definida dentro de outra
função em outro processo). Uma classe comum com __call__, com só
atributos simples (strings/ints), é perfeitamente picklable.

O import do comando real (ReadRTSPCommand/ReadKafkaCommand) é feito DENTRO
do __call__, não no topo do módulo -- assim este arquivo pode ser
importado tanto no processo principal (só para criar a factory, sem
precisar abrir nenhuma câmara) quanto no processo filho (onde o comando é
efetivamente construído).
"""


class RTSPCommandFactory:
    def __init__(self, source: str, width: int, height: int, reconnect_delay: int = 5):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay

    def __call__(self):
        from extraction.frameReaderCommand.ReadRTSPCommand import ReadRTSPCommand
        return ReadRTSPCommand(
            source=self.source,
            width=self.width,
            height=self.height,
            reconnect_delay=self.reconnect_delay,
        )


class KafkaCommandFactory:
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str, width: int, height: int, target_camera_id: str = ""):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.width = width
        self.height = height
        self.target_camera_id = target_camera_id

    def __call__(self):
        from extraction.frameReaderCommand.ReadKafkaCommand import ReadKafkaCommand
        return ReadKafkaCommand(
            bootstrap_servers=self.bootstrap_servers,
            topic=self.topic,
            group_id=self.group_id,
            width=self.width,
            height=self.height,
            target_camera_id=self.target_camera_id,
        )