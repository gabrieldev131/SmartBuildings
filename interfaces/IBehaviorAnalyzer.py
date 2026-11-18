# interfaces/IBehaviorAnalyzer.py
from abc import ABC, abstractmethod
from domain.BehaviorResult import BehaviorResult
from domain.TrackContext import TrackContext


class IBehaviorAnalyzer(ABC):
    """Strategy / Plugin Pattern interface for decoupled behavioral analysis (OCP & ISP)."""

    @classmethod
    def from_config(cls, config) -> "IBehaviorAnalyzer":
        return cls(config)

    @abstractmethod
    def analyze(self, context: TrackContext) -> BehaviorResult:
        pass
