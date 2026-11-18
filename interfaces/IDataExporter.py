# interfaces/IDataExporter.py
from abc import ABC, abstractmethod


class IDataExporter(ABC):
    """Strategy / Sink interface for decoupled tracking data export (OCP, ISP & DIP)."""

    @classmethod
    def from_config(cls, config) -> "IDataExporter":
        return cls()

    def on_event(self, event: dict) -> None:
        pass

    @abstractmethod
    def export(self, records: list[dict]) -> None:
        pass

    def cleanup(self) -> None:
        pass
