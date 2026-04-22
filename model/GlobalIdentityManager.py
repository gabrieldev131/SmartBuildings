# model/GlobalIdentityManager.py
import threading
from scipy.spatial.distance import cosine
import logging

# IMPORTAMOS O NOSSO NOVO TAD
from model.GlobalIdentity import GlobalIdentity

class GlobalIdentityManager:
    def __init__(self, similarity_threshold=0.15):
        # Dicionário agora guarda objetos: { global_id: GlobalIdentity }
        self.identities: dict[int, GlobalIdentity] = {}  
        self.next_global_id = 1
        self.threshold = similarity_threshold
        self._lock = threading.Lock() 

    def get_or_create_global_id(self, new_feature_vector, cam_id: str, current_time: float):
        """
        Identifica a pessoa e atualiza o seu rasto no edifício.
        """
        best_match_id = None
        best_distance = float('inf')

        with self._lock:
            # Procuramos em todos os TADs guardados
            for global_id, identity_obj in self.identities.items():
                # Acedemos à propriedade feature_vector do TAD
                dist = cosine(new_feature_vector, identity_obj.feature_vector)
                
                if dist < best_distance and dist < self.threshold:
                    best_distance = dist
                    best_match_id = global_id

            # Se a pessoa já existe
            if best_match_id is not None:
                # Recuperamos o TAD
                identity = self.identities[best_match_id]
                
                # Avisamos o TAD de onde a pessoa está agora, para ele gerir o histórico
                identity.update_location(cam_id, current_time)
                
                logging.debug(f"Re-ID: ID {best_match_id} em {cam_id} (Dist: {best_distance:.2f})")
                identity.print_summary()
                return best_match_id

            # Se é uma pessoa nova, instanciamos um novo TAD
            new_id = self.next_global_id
            
            nova_identidade = GlobalIdentity(
                global_id=new_id, 
                initial_feature=new_feature_vector, 
                initial_cam_id=cam_id,
                start_time=current_time
            )
            
            self.identities[new_id] = nova_identidade
            self.next_global_id += 1
            
            logging.info(f"Re-ID: Nova identidade registada -> ID {new_id} na {cam_id}")
            return new_id