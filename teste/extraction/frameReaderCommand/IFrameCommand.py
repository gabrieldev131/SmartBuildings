# model/commands/IFrameCommand.py
from abc import ABC, abstractmethod

class IFrameCommand(ABC):
    """
    Interface base para o Padrão Command focado na extração de frames.
    Encapsula a lógica específica de cada fonte de dados.
    """
    
    @abstractmethod
    def execute(self) -> None:
        """
        Executa um ciclo de leitura (um frame ou um batch) e enfileira o resultado.
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Libera os recursos alocados (fecha conexões de rede, descritores de vídeo, etc).
        """
        pass