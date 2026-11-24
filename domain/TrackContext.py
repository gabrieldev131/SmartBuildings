# domain/TrackContext.py
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrackContext:
    """Context passed to each `IBehaviorAnalyzer` plugin with full YOLO, DeepSort, Camera, and Threshold data."""
    global_id: int
    local_id: int
    cam_id: str
    bbox: list[float]
    current_time: float
    confidence: float = 1.0
    object_class: str = "person"
    class_id: int = 0
    people_in_camera: int = 1
    total_tracks_in_camera: int = 1
    confirmed_tracks_in_camera: int = 1
    tentative_tracks_in_camera: int = 0
    coasting_tracks_in_camera: int = 0
    yolo_detections_in_camera: int = 1
    yolo_raw_candidates_in_camera: int = 1
    camera_fps: float = 0.0
    frame_number: int = 0
    frame_width: int = 640
    frame_height: int = 480
    yolo_metrics: dict[str, Any] = field(default_factory=dict)
    deepsort_metrics: dict[str, Any] = field(default_factory=dict)
    limiares_percentuais: dict[str, float] = field(default_factory=dict)

    def camera_summary(self) -> dict[str, Any]:
        occ_pct = self.limiares_percentuais.get("camera_occupancy_pct", 0.0)
        return {
            "camera_id": self.cam_id, "people_in_camera": self.people_in_camera,
            "total_tracks_in_camera": self.total_tracks_in_camera,
            "confirmed_tracks_in_camera": self.confirmed_tracks_in_camera,
            "tentative_tracks_in_camera": self.tentative_tracks_in_camera,
            "coasting_tracks_in_camera": self.coasting_tracks_in_camera,
            "yolo_detections_in_camera": self.yolo_detections_in_camera,
            "yolo_raw_candidates_in_camera": self.yolo_raw_candidates_in_camera,
            "camera_fps": round(self.camera_fps, 2), "frame_number": self.frame_number,
            "frame_width": self.frame_width, "frame_height": self.frame_height,
            "camera_occupancy_pct": occ_pct,
        }

    def to_telemetry_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id, "local_id": self.local_id, "camera_id": self.cam_id,
            "timestamp": round(self.current_time, 3), "object_class": self.object_class,
            "class_id": self.class_id, "people_in_camera": self.people_in_camera,
            "confidence": round(self.confidence, 4),
            "confidence_pct": round(self.confidence * 100.0, 2),
            "bbox_xyxy": [round(float(x), 2) for x in self.bbox],
            "camera_summary": self.camera_summary(),
            "limiares_percentuais": dict(self.limiares_percentuais),
            "yolo": dict(self.yolo_metrics),
            "deepsort": dict(self.deepsort_metrics),
        }
