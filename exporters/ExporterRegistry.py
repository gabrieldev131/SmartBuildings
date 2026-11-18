# exporters/ExporterRegistry.py
import importlib
import pkgutil
from pathlib import Path
from interfaces.IDataExporter import IDataExporter


class ExporterRegistry:
    """Self-registering Plugin Registry Pattern eliminating if/else chains for 100+ exporters (OCP)."""

    _registry: dict[str, type[IDataExporter]] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, name: str):
        def _decorator(exporter_cls: type[IDataExporter]) -> type[IDataExporter]:
            cls._registry[name.lower()] = exporter_cls
            return exporter_cls
        return _decorator

    @classmethod
    def discover_plugins(cls) -> None:
        if cls._discovered:
            return
        cls._discovered = True
        pkg_dir = str(Path(__file__).resolve().parent)
        for _, module_name, _ in pkgutil.iter_modules([pkg_dir]):
            if module_name not in ("ExporterRegistry", "ExportPipeline"):
                importlib.import_module(f"exporters.{module_name}")

    @classmethod
    def create_all(cls, config) -> list[IDataExporter]:
        cls.discover_plugins()
        active_names = getattr(config, "ACTIVE_EXPORTERS", ("csv",))
        return [
            cls._registry[name.lower()].from_config(config)
            for name in active_names
            if name.lower() in cls._registry
        ]
