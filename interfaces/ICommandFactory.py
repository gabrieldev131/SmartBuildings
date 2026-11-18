# interfaces/ICommandFactory.py
from abc import ABC, abstractmethod
from interfaces.IFrameCommand import IFrameCommand


class ICommandFactory(ABC):
    """Abstract Factory Pattern interface for creating picklable IFrameCommand instances."""

    @abstractmethod
    def __call__(self) -> IFrameCommand:
        pass
