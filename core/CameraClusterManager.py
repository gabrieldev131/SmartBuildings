# core/CameraClusterManager.py
import threading


class CameraClusterManager:
    """
    Thread-safe iterative Union-Find algorithm for automatic spatial camera clustering.
    Tracks both cluster roots and full membership sets so the system can report
    which cameras were merged into each Ambiente.
    """

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.merge_history: list[dict] = []
        self._lock = threading.RLock()

    def find(self, cam: str) -> str:
        with self._lock:
            if cam not in self.parent:
                self.parent[cam] = cam
                return cam

            root = cam
            while root != self.parent[root]:
                root = self.parent[root]

            curr = cam
            while curr != root:
                nxt = self.parent[curr]
                self.parent[curr] = root
                curr = nxt

            return root

    def union(self, cam1: str, cam2: str) -> bool:
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

    def get_cluster_members(self, cam: str) -> list[str]:
        """Return all camera IDs currently merged into the same cluster as `cam`."""
        with self._lock:
            target_root = self.find(cam)
            members = [c for c in list(self.parent.keys()) if self.find(c) == target_root]
            return sorted(members) if members else [cam]

    def get_cluster_label(self, cam: str) -> str:
        """Return a '+'-joined string of all cameras in the same cluster as `cam`."""
        return "+".join(self.get_cluster_members(cam))

    def get_all_clusters(self) -> dict[str, list[str]]:
        """Return a mapping {root_cam: [member_cam1, member_cam2, ...]} for all known cameras."""
        with self._lock:
            clusters: dict[str, list[str]] = {}
            for c in list(self.parent.keys()):
                r = self.find(c)
                clusters.setdefault(r, []).append(c)
            for r in clusters:
                clusters[r].sort()
            return clusters
