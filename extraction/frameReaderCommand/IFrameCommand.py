# extraction/frameReaderCommand/IFrameCommand.py
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

class IFrameCommand(ABC):
    """
    Interface base para o Padrão Command focado na extração de frames.
    Agora adaptada para retorno síncrono.
    """
    
    @abstractmethod
    def execute(self) -> Optional[Tuple[str, Any]]:
        """
        Executa um ciclo de leitura e retorna (cam_id, frame) ou None se não houver frame.
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Libera os recursos alocados (fecha conexões de rede, descritores de vídeo, etc).
        """
        pass