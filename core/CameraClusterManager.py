# core/CameraClusterManager.py

class CameraClusterManager:
    """Algoritmo Union-Find para agrupar câmaras automaticamente."""
    def __init__(self):
        self.parent = {}

    def find(self, cam):
        if cam not in self.parent:
            self.parent[cam] = cam
        if self.parent[cam] != cam:
            self.parent[cam] = self.find(self.parent[cam]) # Path compression
        return self.parent[cam]

    def union(self, cam1, cam2):
        root1 = self.find(cam1)
        root2 = self.find(cam2)
        if root1 != root2:
            # Fundimos as raízes (por ordem alfabética para manter o nome previsível)
            if root1 < root2:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
            return True
        return False