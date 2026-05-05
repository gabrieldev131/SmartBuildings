# core/GlobalIdentityManager.py
import threading
import csv
import logging
import numpy as np
from scipy.spatial.distance import cosine

from core.GlobalIdentity import GlobalIdentity

class GlobalIdentityManager:
    def __init__(self, config=None):
        # Permite injetar um config. Se for None, usa parâmetros robustos por defeito
        self.threshold = getattr(config, 'SIMILARITY_THRESHOLD', 0.25)
        self.max_time_lost = getattr(config, 'MAX_TIME_LOST', 5.0)
        self.max_spatial_distance = getattr(config, 'MAX_SPATIAL_DISTANCE', 150)
        
        self.identities: dict[int, GlobalIdentity] = {}  
        self.next_global_id = 1
        self._lock = threading.Lock() 

    def _get_center(self, bbox: list) -> np.ndarray:
        """Calcula o centro da bounding box."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _cleanup_old_identities(self, current_time: float):
        """Remove da memória as pessoas que não são vistas há muito tempo."""
        ids_to_remove = [
            gid for gid, ident in self.identities.items() 
            if (current_time - ident.last_seen) > self.max_time_lost * 2
        ]
        for gid in ids_to_remove:
            del self.identities[gid]

    def get_or_create_global_id(self, new_feature_vector, bbox: list, cam_id: str, current_time: float):
        best_match_id = None
        best_distance = float('inf')
        new_center = self._get_center(bbox)

        with self._lock:
            # 1. Limpeza de memória
            self._cleanup_old_identities(current_time)

            # 2. Iterar identidades ativas
            for global_id, identity_obj in self.identities.items():
                
                # A. Avalia tempo perdido
                time_lost = current_time - identity_obj.last_seen
                if time_lost > self.max_time_lost:
                    continue

                # B. Avalia distância espacial (apenas se for a mesma câmara)
                spatial_dist = 0
                if identity_obj.current_camera == cam_id:
                    old_center = self._get_center(identity_obj.last_bbox)
                    spatial_dist = np.linalg.norm(new_center - old_center)
                    
                    # Se apareceu muito longe num curtíssimo espaço de tempo, não é a mesma pessoa
                    if spatial_dist > self.max_spatial_distance and time_lost < 1.0:
                        continue

                # C. Avalia distância visual
                appearance_dist = cosine(new_feature_vector, identity_obj.feature_vector)
                
                # D. HEURÍSTICA ESPAÇO-TEMPORAL: "Desconto" de similaridade
                # Se estiver muito perto de onde desapareceu há menos de 2 segundos, somos mais tolerantes visualmente.
                if spatial_dist < 50 and time_lost < 2.0:
                    appearance_dist *= 0.5  

                # E. Regista a melhor correspondência
                if appearance_dist < best_distance and appearance_dist < self.threshold:
                    best_distance = appearance_dist
                    best_match_id = global_id

            # 3. Atualizar ou Criar
            if best_match_id is not None:
                identity = self.identities[best_match_id]
                # Nota: A função agora chama-se 'update' no GlobalIdentity
                mudou_de_camera = identity.update(new_feature_vector, bbox, cam_id, current_time)
                
                if mudou_de_camera:
                    logging.info(f"Re-ID Evento: ID {best_match_id} moveu-se para a {cam_id}")
                
                return best_match_id

            # Não encontrou -> Cria nova identidade
            new_id = self.next_global_id
            nova_identidade = GlobalIdentity(
                global_id=new_id, 
                initial_feature=new_feature_vector, 
                bbox=bbox,                  # Novo parâmetro injetado
                initial_cam_id=cam_id,
                start_time=current_time
            )
            
            self.identities[new_id] = nova_identidade
            self.next_global_id += 1
            
            logging.info(f"Re-ID Evento: Nova identidade registada -> ID {new_id} na {cam_id}")
            return new_id
        
    def get_identity_history(self, global_id: int) -> list[dict]:
        with self._lock:
            identity = self.identities.get(global_id)
            if identity:
                return identity.get_human_readable_history()
            return []

    # -------------------------------------------------------------------------
    # EXPORTAÇÃO DE DADOS
    # -------------------------------------------------------------------------
    def export_data_to_csv(self, filename="tracking_data.csv"):
        """
        Exporta todo o histórico de movimentos para um ficheiro CSV.
        Este ficheiro será a base para o algoritmo de Cadeia de Markov.
        """
        with self._lock:
            if not self.identities:
                logging.info("Sem dados para exportar.")
                return

            all_records = []
            for identity in self.identities.values():
                all_records.extend(identity.get_raw_history())

            if not all_records:
                return

            # Extrair o cabeçalho das chaves do primeiro registo
            keys = all_records[0].keys()

            with open(filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(all_records)
                
            logging.info(f"Dados de rastreamento exportados com sucesso para '{filename}'")