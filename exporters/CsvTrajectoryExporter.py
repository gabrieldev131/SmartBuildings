# exporters/CsvTrajectoryExporter.py
import csv
import logging
from exporters.ExporterRegistry import ExporterRegistry
from interfaces.IDataExporter import IDataExporter


@ExporterRegistry.register("csv")
class CsvTrajectoryExporter(IDataExporter):
    """Concrete `IDataExporter` that writes consolidated tracking records to a CSV file (SRP)."""

    CSV_FIELDS = [
        "event_index", "event_type", "global_id", "camera_id", "origin_camera", "next_camera",
        "ambiente_id", "merged_cameras", "cameras_path", "timestamp_in", "timestamp_out",
        "datetime_in", "datetime_out", "duration_seconds", "stopped_seconds", "reid_distance", "details",
    ]

    def __init__(self, filename: str = "tracking_data_final.csv"):
        self.filename = filename

    @classmethod
    def from_config(cls, config) -> "CsvTrajectoryExporter":
        return cls(filename=getattr(config, "EXPORT_CSV_FILENAME", "tracking_data_final.csv"))

    def export(self, records: list[dict]) -> None:
        if not records:
            return
        with open(self.filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        logging.info(f"[CsvTrajectoryExporter] Exportados {len(records)} registos para '{self.filename}'.")
