# core/GlobalIdentityManager.py
import csv
import time
import logging
import threading
import numpy as np
from datetime import datetime

from core.GlobalIdentity import GlobalIdentity
from core.CameraClusterManager import CameraClusterManager

class GlobalIdentityManager:
    """
    Estado compartilhado entre TODAS as câmaras/threads de processamento.
    É essa mesma instância (mesmo objeto em memória) que é passada para
    cada CameraWorker -- é isso que garante que uma pessoa vista em
    câmaras diferentes recebe o mesmo global_id.

    Como mais de uma thread pode chamar get_or_create_global_id /
    update_existing_identity ao mesmo tempo (uma thread de processamento
    por câmara, por exemplo), todo acesso ao dicionário self.identities
    (leitura+escrita, "check-then-act") precisa ser atômico. self._lock
    garante isso: cada chamada pública que lê e depois decide escrever
    é protegida do início ao fim, então duas câmaras nunca criam dois
    IDs novos para a mesma pessoa numa corrida, nem corrompem o dicionário
    sendo alterado por duas threads ao mesmo tempo.
    """
    def __init__(self, config=None):
        self.config = config
        
        self.intra_camera_threshold = getattr(config, 'SIMILARITY_THRESHOLD', 0.25) 
        self.inter_camera_threshold = getattr(config, 'INTER_CAMERA_THRESHOLD', 0.40)
        
        self.max_time_lost = getattr(config, 'MAX_TIME_LOST', 10.0)
        
        # Guardamos o quadrado do limite para evitar raízes quadradas depois
        max_dist = getattr(config, 'MAX_SPATIAL_DISTANCE', 150)
        self.max_spatial_distance_sq = max_dist ** 2
        
        self.switch_cooldown = getattr(config, 'CAMERA_SWITCH_COOLDOWN', 2.0)
        self.cluster_manager = CameraClusterManager()
        self.auto_cluster_threshold = getattr(config, 'AUTO_CLUSTER_TIME_THRESHOLD', 3.0)
        
        self.frame_w = getattr(config, 'PROCESSING_WIDTH', 640)
        self.frame_h = getattr(config, 'PROCESSING_HEIGHT', 480)
        
        self.identities: dict[int, GlobalIdentity] = {}  
        self.next_global_id = 1

        # Lock único protegendo self.identities / self.next_global_id.
        # Reentrante (RLock) porque os métodos públicos chamam métodos
        # internos (_check_and_update_clusters, _cleanup_old_identities)
        # que também podem precisar do lock -- com RLock a mesma thread
        # pode adquiri-lo de novo sem se travar.
        self._lock = threading.RLock()

    def _get_center(self, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _is_near_edge(self, bbox: list) -> bool:
        x1, y1, x2, y2 = bbox
        margin_x = self.frame_w * 0.05 
        margin_y = self.frame_h * 0.05 
        return (x1 < margin_x) or (y1 < margin_y) or (x2 > self.frame_w - margin_x) or (y2 > self.frame_h - margin_y)

    def _cleanup_old_identities(self, current_time: float):
        ids_to_remove = [
            gid for gid, ident in self.identities.items() 
            if (current_time - ident.last_seen) > self.max_time_lost * 2
        ]
        for gid in ids_to_remove:
            del self.identities[gid]

    def _check_and_update_clusters(self, identity_obj: GlobalIdentity, cam_id: str, current_time: float):
        for old_cam, last_t in identity_obj.last_seen_per_camera.items():
            if old_cam != cam_id and (current_time - last_t) <= self.auto_cluster_threshold:
                if self.cluster_manager.union(cam_id, old_cam):
                    root_cluster = self.cluster_manager.find(cam_id)
                    logging.info(f" Mapeamento: Câmaras '{cam_id}' e '{old_cam}' fundidas no 'Ambiente_{root_cluster}'.")

    def update_existing_identity(self, global_id: int, new_feature_vector, bbox: list, cam_id: str, current_time: float):
        with self._lock:
            if global_id in self.identities:
                identity = self.identities[global_id]
                self._check_and_update_clusters(identity, cam_id, current_time)
                identity.update(new_feature_vector, bbox, cam_id, current_time, switch_cooldown=self.switch_cooldown)

    def get_or_create_global_id(self, new_feature_vector, bbox: list, cam_id: str, current_time: float, active_global_ids: set = None):
        if active_global_ids is None:
            active_global_ids = set()
            
        best_match_id = None
        best_distance = float('inf')
        new_center = self._get_center(bbox)

        with self._lock:
            self._cleanup_old_identities(current_time)

            for global_id, identity_obj in self.identities.items():

                if global_id in active_global_ids:
                    continue

                time_lost = current_time - identity_obj.last_seen

                if time_lost > self.max_time_lost:
                    continue

                dot_product = np.dot(new_feature_vector, identity_obj.feature_vector)
                appearance_dist = max(0.0, 1.0 - dot_product)

                is_same_camera = (identity_obj.current_camera == cam_id)

                if is_same_camera:
                    old_center = self._get_center(identity_obj.last_bbox)

                    # CÁLCULO OTIMIZADO: Distância ao quadrado (sem np.sqrt)
                    spatial_dist_sq = (new_center[0] - old_center[0])**2 + (new_center[1] - old_center[1])**2

                    # Prevenção de Teletransporte comparando com o limite ao quadrado
                    if spatial_dist_sq > self.max_spatial_distance_sq and time_lost < 1.0:
                        continue

                    exited_scene = self._is_near_edge(identity_obj.last_bbox)
                    is_completely_different = appearance_dist > 0.50

                    if not exited_scene and not is_completely_different:
                        # 80^2 = 6400 | 150^2 = 22500
                        if spatial_dist_sq < 6400 and time_lost < 2.0:
                            appearance_dist *= 0.1
                        elif spatial_dist_sq < 22500 and time_lost < 4.0:
                            appearance_dist *= 0.4

                    if appearance_dist < best_distance and appearance_dist < self.intra_camera_threshold:
                        best_distance = appearance_dist
                        best_match_id = global_id

                else:
                    if appearance_dist < best_distance and appearance_dist < self.inter_camera_threshold:
                        best_distance = appearance_dist
                        best_match_id = global_id

            if best_match_id is not None:
                identity = self.identities[best_match_id]
                self._check_and_update_clusters(identity, cam_id, current_time)

                mudou_de_camera = identity.update(new_feature_vector, bbox, cam_id, current_time, self.switch_cooldown)
                if mudou_de_camera:
                    logging.info(f"Re-ID Evento: ID {best_match_id} moveu-se permanentemente para a {cam_id} (Distância: {best_distance:.2f})")

                return best_match_id

            new_id = self.next_global_id
            nova_identidade = GlobalIdentity(
                global_id=new_id,
                initial_feature=new_feature_vector,
                bbox=bbox,
                initial_cam_id=cam_id,
                start_time=current_time
            )

            self.identities[new_id] = nova_identidade
            self.next_global_id += 1
            logging.info(f"Re-ID Evento: Nova pessoa -> ID {new_id} na {cam_id}")
            return new_id

    def _aggregate_history_by_clusters(self, global_id: int, raw_history: list) -> list[dict]:
        if not raw_history:
            return []
            
        aggregated = []
        current_cluster = self.cluster_manager.find(raw_history[0]["camera_id"])
        t_in = raw_history[0]["timestamp_in"]
        t_out = raw_history[0]["timestamp_out"]
        
        for record in raw_history[1:]:
            cluster = self.cluster_manager.find(record["camera_id"])
            r_in = record["timestamp_in"]
            r_out = record["timestamp_out"]
            
            if cluster == current_cluster:
                t_out = max(t_out, r_out)
            else:
                aggregated.append({
                    "global_id": global_id,
                    "ambiente_id": f"Ambiente_{current_cluster}",
                    "timestamp_in": t_in,
                    "timestamp_out": t_out,
                    "duration_seconds": t_out - t_in
                })
                current_cluster = cluster
                t_in = r_in
                t_out = r_out
                
        aggregated.append({
            "global_id": global_id,
            "ambiente_id": f"Ambiente_{current_cluster}",
            "timestamp_in": t_in,
            "timestamp_out": t_out,
            "duration_seconds": t_out - t_in
        })
        return aggregated

    def _format_timestamp(self, timestamp: float) -> str:
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object.strftime("%H:%M:%S:%d/%m/%Y")

    def get_identity_history(self, global_id: int) -> list[dict]:
        with self._lock:
            identity = self.identities.get(global_id)
            if not identity:
                return []
            raw = identity.get_raw_history()

        clustered = self._aggregate_history_by_clusters(global_id, raw)
        
        formatted = []
        for record in clustered:
            formatted.append({
                "Ambiente": record["ambiente_id"],
                "Entrada": self._format_timestamp(record["timestamp_in"]),
                "Saida": self._format_timestamp(record["timestamp_out"])
            })
        return formatted

    def export_data_to_csv(self, filename="tracking_data.csv"):

        with self._lock:
            if not self.identities:
                return

            all_records = []
            for gid, identity in self.identities.items():
                raw = identity.get_raw_history()
                clustered = self._aggregate_history_by_clusters(gid, raw)
                all_records.extend(clustered)

        if not all_records:
            return
        keys = all_records[0].keys()

        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_records)