# view/console_view.py
import cv2

class ConsoleView:
    def display_message(self, message):
        print(message)
    
    # ... (os outros métodos de display, como display_scan_start, continuam iguais) ...
    def display_scan_start(self, network_range, port):
        print(f"Iniciando varredura na rede {network_range} pela porta {port}...")

    def display_camera_found(self, ip):
        print(f"  [+] Câmera encontrada em: {ip}")

    def display_scan_complete(self, count):
        print(f"\nVarredura concluída. {count} câmera(s) encontrada(s).")
    
    # --- NOVOS MÉTODOS PARA O VÍDEO AO VIVO ---
    
    def show_frame(self, window_name, frame):
        """Exibe um frame em uma janela do OpenCV."""
        if frame is not None:
            cv2.imshow(window_name, frame)

    def get_keypress(self):
        """Aguarda por uma tecla pressionada por 1ms. Essencial para as janelas atualizarem."""
        # O valor '1' é importante para não travar o loop esperando uma tecla.
        return cv2.waitKey(1) & 0xFF

    def destroy_windows(self):
        """Fecha todas as janelas abertas pelo OpenCV."""
        cv2.destroyAllWindows()