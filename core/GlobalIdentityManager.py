# core/GlobalIdentityManager.py
import threading
import csv
import logging
import numpy as np
from scipy.spatial.distance import cosine

from core.GlobalIdentity import GlobalIdentity

class GlobalIdentityManager:
    def __init__(self, config=None):
        self.threshold = getattr(config, 'SIMILARITY_THRESHOLD', 0.25) 
        self.max_time_lost = getattr(config, 'MAX_TIME_LOST', 5.0)
        self.max_spatial_distance = getattr(config, 'MAX_SPATIAL_DISTANCE', 150)
        self.switch_cooldown = getattr(config, 'CAMERA_SWITCH_COOLDOWN', 2.0)
        
        # Dimensões de processamento da câmara (Padrão 640x480)
        self.frame_w = getattr(config, 'PROCESSING_WIDTH', 640)
        self.frame_h = getattr(config, 'PROCESSING_HEIGHT', 480)
        
        self.identities: dict[int, GlobalIdentity] = {}  
        self.next_global_id = 1
        self._lock = threading.Lock() 

    def _get_center(self, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _is_near_edge(self, bbox: list) -> bool:
        """
        Verifica se a Bounding Box está colada a uma das bordas do ecrã.
        Indica que a pessoa provavelmente saiu do campo de visão (FOV).
        """
        x1, y1, x2, y2 = bbox
        margin_x = self.frame_w * 0.05  # 5% da largura (~32px)
        margin_y = self.frame_h * 0.05  # 5% da altura (~24px)
        
        return (x1 < margin_x) or (y1 < margin_y) or (x2 > self.frame_w - margin_x) or (y2 > self.frame_h - margin_y)

    def _cleanup_old_identities(self, current_time: float):
        ids_to_remove = [
            gid for gid, ident in self.identities.items() 
            if (current_time - ident.last_seen) > self.max_time_lost * 2
        ]
        for gid in ids_to_remove:
            del self.identities[gid]

    def update_existing_identity(self, global_id: int, new_feature_vector, bbox: list, cam_id: str, current_time: float):
        with self._lock:
            if global_id in self.identities:
                identity = self.identities[global_id]
                mudou_de_camera = identity.update(
                    new_feature_vector, 
                    bbox, 
                    cam_id, 
                    current_time, 
                    switch_cooldown=self.switch_cooldown
                )
                if mudou_de_camera:
                    logging.info(f"Re-ID: ID {global_id} assumido pela {cam_id} (Handoff contínuo)")

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

                spatial_dist = 0
                if identity_obj.current_camera == cam_id:
                    old_center = self._get_center(identity_obj.last_bbox)
                    spatial_dist = np.linalg.norm(new_center - old_center)
                    
                    if spatial_dist > self.max_spatial_distance and time_lost < 1.0:
                        continue

                appearance_dist = cosine(new_feature_vector, identity_obj.feature_vector)
                
                # REFINAMENTO 1: Prevenção do Problema da Porta (Edge Exit)
                # Se a pessoa sumiu colada à borda do ecrã, é provável que tenha saído de cena.
                exited_scene = self._is_near_edge(identity_obj.last_bbox)
                
                # REFINAMENTO 2: Visual Sanity Check
                # Se as roupas forem ABSURDAMENTE diferentes (distância alta),
                # vetamos a união, impedindo trocas de ID mesmo se ocorrerem no meio do ecrã.
                is_completely_different = appearance_dist > 0.50

                # Só aplicamos a Recuperação Fantasma se a pessoa NÃO tiver saído da cena 
                # e se as roupas não forem os exatos opostos.
                if not exited_scene and not is_completely_different:
                    if spatial_dist < 80 and time_lost < 2.0:
                        appearance_dist *= 0.1  # Confiança alta (Oclusão breve no meio do ecrã)
                    elif spatial_dist < 150 and time_lost < 4.0:
                        appearance_dist *= 0.4  # Confiança média (Oclusão mais longa)

                if appearance_dist < best_distance and appearance_dist < self.threshold:
                    best_distance = appearance_dist
                    best_match_id = global_id

            if best_match_id is not None:
                identity = self.identities[best_match_id]
                identity.update(new_feature_vector, bbox, cam_id, current_time, self.switch_cooldown)
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
            return new_id
            
    def get_identity_history(self, global_id: int) -> list[dict]:
        with self._lock:
            identity = self.identities.get(global_id)
            if identity:
                return identity.get_human_readable_history()
            return []

    def export_data_to_csv(self, filename="tracking_data.csv"):
        with self._lock:
            if not self.identities:
                return

            all_records = []
            for identity in self.identities.values():
                all_records.extend(identity.get_raw_history())

            if not all_records:
                return
            keys = all_records[0].keys()

            with open(filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(all_records)