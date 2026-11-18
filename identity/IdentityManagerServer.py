# identity/IdentityManagerServer.py
from multiprocessing.managers import BaseManager
from identity.GlobalIdentityManager import GlobalIdentityManager


class IdentityManagerServer(BaseManager):
    """Dedicated IPC RPC server hosting the shared `GlobalIdentityManager` in multiprocess mode."""
    pass


IdentityManagerServer.register("GlobalIdentityManager", GlobalIdentityManager)
