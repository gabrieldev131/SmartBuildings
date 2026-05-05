# main.py
import threading
import cv2

from Config import Config
from core.GlobalIdentityManager import GlobalIdentityManager  # O seu ficheiro existente
from vision.cameraWorker import process_camera_stream
from extraction.frameReaderCommand.ReadKafkaCommand import ReadKafkaCommand
from extraction.frameReaderCommand.ReadRTSPCommand import ReadRTSPCommand


def main():
    config = Config()
    
    CAMERAS = {
        "Cam_Rua": "examples/pessoas_rua_60fps.mp4",
        # "Cam_Entrada": 0  # Exemplo webcam
    }

    global_manager = GlobalIdentityManager(config)
    threads = []
    
    for cam_id, source in CAMERAS.items():
        thread = threading.Thread(
            target=process_camera_stream, 
            args=(cam_id, source, config, global_manager),
            daemon=True
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    global_manager.export_data_to_csv("dados_rastreamento.csv")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()