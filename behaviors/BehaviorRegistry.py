# behaviors/BehaviorRegistry.py
import importlib
import pkgutil
from pathlib import Path
from interfaces.IBehaviorAnalyzer import IBehaviorAnalyzer


class BehaviorRegistry:
    """Self-registering Plugin Registry Pattern eliminating if/else chains for behavior plugins (OCP)."""

    _registry: dict[str, type[IBehaviorAnalyzer]] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, name: str):
        def _decorator(plugin_cls: type[IBehaviorAnalyzer]) -> type[IBehaviorAnalyzer]:
            cls._registry[name.lower()] = plugin_cls
            return plugin_cls
        return _decorator

    @classmethod
    def discover_plugins(cls) -> None:
        if cls._discovered:
            return
        cls._discovered = True
        pkg_dir = str(Path(__file__).resolve().parent)
        for _, module_name, _ in pkgutil.iter_modules([pkg_dir]):
            if module_name not in ("BehaviorRegistry", "BehaviorPipeline"):
                importlib.import_module(f"behaviors.{module_name}")

    @classmethod
    def create_all(cls, config) -> list[IBehaviorAnalyzer]:
        cls.discover_plugins()
        active_names = getattr(config, "ACTIVE_BEHAVIORS", ("stopped",))
        return [
            cls._registry[name.lower()].from_config(config)
            for name in active_names
            if name.lower() in cls._registry
        ]
