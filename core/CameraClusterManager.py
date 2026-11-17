# core/CameraClusterManager.py
"""
Camera clustering manager built on SciPy's `DisjointSet` (Union-Find).
Automatically groups cameras into shared spatial environments ("Ambientes")
when people transition between them within a short time window.
"""
import threading
from scipy.cluster.hierarchy import DisjointSet


class CameraClusterManager:
    """Thread-safe wrapper around `scipy.cluster.hierarchy.DisjointSet` for camera topology."""

    def __init__(self):
        self._disjoint_set = DisjointSet()
        self._lock = threading.RLock()

    def _ensure_added(self, cam: str) -> None:
        if cam not in self._disjoint_set:
            self._disjoint_set.add(cam)

    def find(self, cam: str) -> str:
        """Return the canonical root camera ID for the cluster containing `cam`."""
        with self._lock:
            self._ensure_added(cam)
            return min(self._disjoint_set.subset(cam))

    def union(self, cam1: str, cam2: str) -> bool:
        """Merge the clusters of `cam1` and `cam2`. Returns True if a new merge occurred."""
        with self._lock:
            self._ensure_added(cam1)
            self._ensure_added(cam2)
            if self._disjoint_set.connected(cam1, cam2):
                return False
            self._disjoint_set.merge(cam1, cam2)
            return True

    def get_cluster_members(self, cam: str) -> list[str]:
        """Return sorted list of all cameras in the same cluster as `cam`."""
        with self._lock:
            self._ensure_added(cam)
            return sorted(self._disjoint_set.subset(cam))

    def get_cluster_label(self, cam: str) -> str:
        """Return a '+'-joined string of all cameras in the same cluster as `cam`."""
        return "+".join(self.get_cluster_members(cam))

    def get_all_clusters(self) -> dict[str, list[str]]:
        """Return mapping `{root_cam: [member_cams...]}` for all registered cameras."""
        with self._lock:
            return {min(subset): sorted(subset) for subset in self._disjoint_set.subsets()}
