# model/GlobalIdentityManager.py
import threading
import csv
import logging
from scipy.spatial.distance import cosine

from model.GlobalIdentity import GlobalIdentity

class GlobalIdentityManager:
    def __init__(self, similarity_threshold=0.15):
        self.identities: dict[int, GlobalIdentity] = {}  
        self.next_global_id = 1
        self.threshold = similarity_threshold
        self._lock = threading.Lock() 

    def get_or_create_global_id(self, new_feature_vector, cam_id: str, current_time: float):
        best_match_id = None
        best_distance = float('inf')

        with self._lock:
            for global_id, identity_obj in self.identities.items():
                dist = cosine(new_feature_vector, identity_obj.feature_vector)
                
                if dist < best_distance and dist < self.threshold:
                    best_distance = dist
                    best_match_id = global_id

            if best_match_id is not None:
                identity = self.identities[best_match_id]
                mudou_de_camera = identity.update_location(cam_id, current_time)
                
                if mudou_de_camera:
                    logging.info(f"Re-ID Evento: ID {best_match_id} moveu-se para a {cam_id}")
                
                return best_match_id

            new_id = self.next_global_id
            nova_identidade = GlobalIdentity(
                global_id=new_id, 
                initial_feature=new_feature_vector, 
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
    # EXPORTAÇÃO DE DADOS (NOVO)
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