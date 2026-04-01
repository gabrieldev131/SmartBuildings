# model/GlobalIdentityManager.py
import numpy as np
import threading
from scipy.spatial.distance import cosine
import logging

class GlobalIdentityManager:
    # 1. Limiar mais rigoroso (0.15 em vez de 0.4)
    def __init__(self, similarity_threshold=0.15):
        self.identities = {}  
        self.next_global_id = 1
        self.threshold = similarity_threshold
        self._lock = threading.Lock() 

    def get_or_create_global_id(self, new_feature_vector):
        best_match_id = None
        best_distance = float('inf')

        with self._lock:
            for global_id, saved_vector in self.identities.items():
                dist = cosine(new_feature_vector, saved_vector)
                
                if dist < best_distance and dist < self.threshold:
                    best_distance = dist
                    best_match_id = global_id

            if best_match_id is not None:
                logging.debug(f"Re-ID Global: ID {best_match_id} reconhecido (Dist: {best_distance:.2f})")
                
                # 2. REMOVIDA A FUSÃO DE VETORES (EMA)
                # Mantemos a assinatura original pura para evitar o Colapso de Identidade
                
                return best_match_id

            new_id = self.next_global_id
            self.identities[new_id] = np.array(new_feature_vector)
            logging.info(f"Re-ID Global: Nova identidade registada -> ID Global {new_id}")
            self.next_global_id += 1
            
            return new_id