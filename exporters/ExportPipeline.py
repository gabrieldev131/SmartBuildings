# exporters/ExportPipeline.py
from datetime import datetime
from typing import Optional
from exporters.ExporterRegistry import ExporterRegistry
from interfaces.IDataExporter import IDataExporter


class ExportPipeline:
    """Composite exporter that formats trajectories and dispatches to registered `IDataExporter`s."""

    def __init__(self, cluster_manager, config=None, exporters: Optional[list[IDataExporter]] = None):
        self.cluster_manager = cluster_manager
        self.config = config
        self.exporters: list[IDataExporter] = (
            list(exporters) if exporters is not None else ExporterRegistry.create_all(config)
        )

    def register(self, exporter: IDataExporter) -> None:
        self.exporters.append(exporter)

    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def notify_event(self, event: dict) -> None:
        for exporter in self.exporters:
            exporter.on_event(event)

    def _make_summary_row(self, gid: int, cluster: str, origin: str, next_env: str, cams: list[str], t_in: float, t_out: float, stopped: float) -> dict:
        merged = self.cluster_manager.get_cluster_label(cluster)
        path_str, dur = " -> ".join(cams), max(0.0, t_out - t_in)
        return {
            "event_type": "AMBIENTE_SUMMARY", "global_id": gid, "camera_id": cluster,
            "origin_camera": origin, "next_camera": next_env, "ambiente_id": f"Ambiente_{cluster}",
            "merged_cameras": merged, "cameras_path": path_str, "timestamp_in": t_in,
            "timestamp_out": t_out, "duration_seconds": dur, "stopped_seconds": stopped,
            "reid_distance": 0.0, "reid_similarity_pct": 100.0, "people_in_camera": 1,
            "object_class": "person", "class_id": 0, "confidence_pct": 100.0,
            "details": f"Resumo Ambiente: ID {gid} em 'Ambiente_{cluster}' por {dur:.2f}s",
        }

    def aggregate_by_clusters(self, global_id: int, raw_history: list[dict]) -> list[dict]:
        if not raw_history:
            return []
        summaries, cams = [], [raw_history[0]["camera_id"]]
        curr = self.cluster_manager.find(cams[0])
        t_in, t_out, stop_sum, origin = raw_history[0]["timestamp_in"], raw_history[0]["timestamp_out"], float(raw_history[0].get("stopped_seconds", 0.0)), "Entrada"
        for rec in raw_history[1:]:
            cl = self.cluster_manager.find(rec["camera_id"])
            if cl == curr:
                t_out, stop_sum = max(t_out, rec["timestamp_out"]), stop_sum + float(rec.get("stopped_seconds", 0.0))
                if cams[-1] != rec["camera_id"]:
                    cams.append(rec["camera_id"])
            else:
                summaries.append(self._make_summary_row(global_id, curr, origin, f"Ambiente_{cl}", cams, t_in, t_out, stop_sum))
                origin, curr, t_in, t_out, stop_sum, cams = f"Ambiente_{curr}", cl, rec["timestamp_in"], rec["timestamp_out"], float(rec.get("stopped_seconds", 0.0)), [rec["camera_id"]]
        summaries.append(self._make_summary_row(global_id, curr, origin, "Saida", cams, t_in, t_out, stop_sum))
        return summaries

    def _build_event_row(self, ev: dict, all_identities: dict) -> dict:
        cam_id, gid = ev.get("camera_id", ""), ev.get("global_id", "")
        root = self.cluster_manager.find(cam_id) if cam_id else ""
        merged = self.cluster_manager.get_cluster_label(cam_id) if cam_id else ""
        ident = all_identities.get(gid)
        path = " -> ".join(r["camera_id"] for r in ident.get_raw_history()) if ident else ev.get("cameras_path", cam_id)
        dist = float(ev.get("reid_distance", 0.0))
        return {
            **ev, "event_type": ev["event_type"], "global_id": gid, "camera_id": cam_id,
            "origin_camera": ev.get("origin_camera", ""), "next_camera": ev.get("next_camera", ""),
            "ambiente_id": ev.get("ambiente_id") or (f"Ambiente_{root}" if root else ""),
            "merged_cameras": ev.get("merged_cameras") or merged, "cameras_path": ev.get("cameras_path") or path,
            "timestamp_in": float(ev["timestamp_in"]), "timestamp_out": float(ev["timestamp_out"]),
            "duration_seconds": float(ev.get("duration_seconds", 0.0)), "stopped_seconds": float(ev.get("stopped_seconds", 0.0)),
            "reid_distance": dist, "reid_similarity_pct": float(ev.get("reid_similarity_pct", round(max(0.0, 1.0 - dist) * 100.0, 2))),
            "people_in_camera": int(ev.get("people_in_camera", 1)), "object_class": str(ev.get("object_class", "person")),
            "confidence_pct": float(ev.get("confidence_pct", 100.0)), "details": ev.get("details", ""),
        }

    def _build_passage_row(self, gid: int, rec: dict, full_path: str) -> dict:
        cam_id = rec["camera_id"]
        root, merged = self.cluster_manager.find(cam_id), self.cluster_manager.get_cluster_label(cam_id)
        dur, dist = float(rec["duration_seconds"]), float(rec["reid_distance"])
        return {
            "event_type": "CAMERA_PASSAGE", "global_id": gid, "camera_id": cam_id,
            "origin_camera": rec["origin_camera"], "next_camera": rec["next_camera"],
            "ambiente_id": f"Ambiente_{root}", "merged_cameras": merged, "cameras_path": full_path,
            "timestamp_in": float(rec["timestamp_in"]), "timestamp_out": float(rec["timestamp_out"]),
            "duration_seconds": dur, "stopped_seconds": float(rec["stopped_seconds"]),
            "reid_distance": dist, "reid_similarity_pct": round(max(0.0, 1.0 - dist) * 100.0, 2),
            "people_in_camera": 1, "object_class": "person", "class_id": 0, "confidence_pct": 100.0,
            "details": f"Passagem: ID {gid} em '{cam_id}' ({dur:.2f}s) | Trajeto: [{full_path}]",
        }

    def _format_record(self, idx: int, r: dict) -> dict:
        t_in, t_out = float(r["timestamp_in"]), float(r["timestamp_out"])
        dist = float(r.get("reid_distance", 0.0))
        return {
            **r, "event_index": idx, "event_type": r["event_type"], "global_id": r["global_id"],
            "camera_id": r["camera_id"], "origin_camera": r["origin_camera"], "next_camera": r["next_camera"],
            "ambiente_id": r["ambiente_id"], "merged_cameras": r["merged_cameras"], "cameras_path": r["cameras_path"],
            "people_in_camera": int(r.get("people_in_camera", 1)), "object_class": str(r.get("object_class", "person")),
            "confidence_pct": f"{float(r.get('confidence_pct', 100.0)):.2f}",
            "reid_similarity_pct": f"{float(r.get('reid_similarity_pct', max(0.0, 1.0 - dist) * 100.0)):.2f}",
            "timestamp_in": f"{t_in:.3f}", "timestamp_out": f"{t_out:.3f}",
            "datetime_in": self.format_timestamp(t_in), "datetime_out": self.format_timestamp(t_out),
            "duration_seconds": f"{float(r['duration_seconds']):.3f}", "stopped_seconds": f"{float(r['stopped_seconds']):.3f}",
            "reid_distance": f"{dist:.4f}", "details": r["details"],
        }

    def build_records(self, all_identities: dict, events_copy: list[dict]) -> list[dict]:
        if not all_identities and not events_copy:
            return []
        rows = [self._build_event_row(ev, all_identities) for ev in events_copy]
        for gid in sorted(all_identities.keys()):
            raw = all_identities[gid].get_raw_history()
            full_path = " -> ".join(r["camera_id"] for r in raw)
            rows.extend(self._build_passage_row(gid, rec, full_path) for rec in raw)
            rows.extend(self.aggregate_by_clusters(gid, raw))
        rows.sort(key=lambda r: (1 if r["event_type"] == "AMBIENTE_SUMMARY" else 0, r["timestamp_in"]))
        return [self._format_record(i, r) for i, r in enumerate(rows, start=1)]

    def export_all(self, all_identities: dict, events_copy: list[dict]) -> None:
        records = self.build_records(all_identities, events_copy)
        if not records:
            return
        for exporter in self.exporters:
            exporter.export(records)
            exporter.cleanup()
