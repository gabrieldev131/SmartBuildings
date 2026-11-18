# identity/CameraClusterManager.py
import threading
from scipy.cluster.hierarchy import DisjointSet


class CameraClusterManager:
    """Thread-safe wrapper around `scipy.cluster.hierarchy.DisjointSet` for camera topology (SRP)."""

    def __init__(self):
        self._disjoint_set = DisjointSet()
        self._lock = threading.RLock()

    def _ensure_added(self, cam: str) -> None:
        if cam not in self._disjoint_set:
            self._disjoint_set.add(cam)

    def find(self, cam: str) -> str:
        with self._lock:
            self._ensure_added(cam)
            return min(self._disjoint_set.subset(cam))

    def union(self, cam1: str, cam2: str) -> bool:
        with self._lock:
            self._ensure_added(cam1)
            self._ensure_added(cam2)
            if self._disjoint_set.connected(cam1, cam2):
                return False
            self._disjoint_set.merge(cam1, cam2)
            return True

    def get_cluster_members(self, cam: str) -> list[str]:
        with self._lock:
            self._ensure_added(cam)
            return sorted(self._disjoint_set.subset(cam))

    def get_cluster_label(self, cam: str) -> str:
        return "+".join(self.get_cluster_members(cam))

    def get_all_clusters(self) -> dict[str, list[str]]:
        with self._lock:
            return {min(subset): sorted(subset) for subset in self._disjoint_set.subsets()}
