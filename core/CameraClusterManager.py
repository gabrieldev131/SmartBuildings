# core/CameraClusterManager.py
import threading


class CameraClusterManager:
    """
    Algoritmo Union-Find iterativo para agrupar câmaras automaticamente.

    Tem o seu próprio lock (independente do lock do GlobalIdentityManager)
    porque self.parent também é lido em _aggregate_history_by_clusters,
    que roda fora da seção crítica do GlobalIdentityManager (ver
    get_identity_history / export_data_to_csv). Um RLock simples aqui é
    suficiente: as operações são rápidas (find/union) e não chamam de volta
    nenhum método do GlobalIdentityManager.
    """
    def __init__(self):
        self.parent = {}
        self._lock = threading.RLock()

    def find(self, cam):
        with self._lock:
            if cam not in self.parent:
                self.parent[cam] = cam
                return cam

            # 1. Encontra a raiz de forma iterativa (não recursiva)
            root = cam
            while root != self.parent[root]:
                root = self.parent[root]

            # 2. Path compression iterativo (achata a árvore para buscas futuras em O(1))
            curr = cam
            while curr != root:
                nxt = self.parent[curr]
                self.parent[curr] = root
                curr = nxt

            return root

    def union(self, cam1, cam2):
        with self._lock:
            root1 = self.find(cam1)
            root2 = self.find(cam2)
            if root1 != root2:
                if root1 < root2:
                    self.parent[root2] = root1
                else:
                    self.parent[root1] = root2
                return True
            return False