# model/network_scanner.py
import socket

class NetworkScanner():
    def __init__(self, network_base, ip_range, port):
        self._network_base = network_base
        self._ip_range = ip_range
        self._port = port

    def discover(self, progress_callback=None):
        """Escaneia a rede e retorna uma lista de IPs de câmeras encontradas."""
        found_ips = []
        for i in self._ip_range:
            ip = f"{self._network_base}.{i}"
            try:
                with socket.create_connection((ip, self._port), timeout=0.1):
                    found_ips.append(ip)
                    if progress_callback:
                        progress_callback(ip) # Informa o progresso em tempo real
            except (socket.timeout, ConnectionRefusedError):
                continue
        return found_ips