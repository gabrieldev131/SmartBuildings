# controller/app_controller.py
import threading
import queue
import time
import logging
import cv2

# Importação dos seus comandos (Ajuste os imports conforme a sua estrutura)
from extraction.frameReaderCommand.ReadRTSPCommand import ReadRTSPCommand
from extraction.frameReaderCommand.ReadKafkaCommand import ReadKafkaCommand
from extraction.frameReaderCommand.FrameReaderInvoker import FrameReaderInvoker

from core.GlobalIdentityManager import GlobalIdentityManager
from vision.cameraWorker import CameraWorker

class FrameReaderManagement:
    """
    Orquestrador que liga o Padrão Command (Aquisição) ao Processamento (Workers).
    """
    def __init__(self, config):
        self.config = config
        self.stop_event = threading.Event()
        
        # Fila onde os Comandos colocam as imagens brutas
        self.raw_frames_queue = queue.Queue(maxsize=100)
        
        # Dicionário de filas e workers por câmara
        self.camera_queues = {}
        self.camera_workers = {}
        
        # Manager de Identidades Único
        self.global_id_manager = GlobalIdentityManager(config)
        self._reader_invoker = None

    def run(self):
        logging.info("A iniciar o sistema...")
        self._start_frame_reader()
        self._main_routing_loop()
        self._shutdown()

    def _start_frame_reader(self):
        """Inicializa a abstração de extração de frames (Command Pattern)."""
        
        # Exemplo com RTSP (Poderia ser o ReadKafkaCommand)
        video_command = ReadRTSPCommand(
            source="examples/pessoas_rua_60fps.mp4", 
            output_queue=self.raw_frames_queue,
            width=640,
            height=480
        )
        
        self._reader_invoker = FrameReaderInvoker(
            command=video_command,
            stop_event=self.stop_event,
            name="VideoReaderInvoker"
        )

        """kafka_command = ReadKafkaCommand(
            output_queue=self.raw_frames_queue,
            bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS,
            topic=self.config.KAFKA_TOPIC,
            group_id=self.config.KAFKA_GROUP_ID,
            width=self.config.PROCESSING_WIDTH,
            height=self.config.PROCESSING_HEIGHT,
            target_camera_id=getattr(self.config, "KAFKA_TARGET_CAMERA", None)
        )

        self._reader_invoker = FrameReaderInvoker(
            command=kafka_command,
            stop_event=self.stop_event,
            name="VideoReaderInvoker"
        )"""
        self._reader_invoker.start()

    def _main_routing_loop(self):
        """
        Lê a fila partilhada e distribui os frames para as threads 
        de processamento corretas de forma dinâmica.
        """
        logging.info("Router principal ativo. A aguardar frames...")
        
        while not self.stop_event.is_set():
            try:
                # Tenta pegar um frame gerado pelos Comandos
                cam_id, frame = self.raw_frames_queue.get(timeout=0.5)

                # Descoberta Dinâmica de Câmaras (Dynamic Provisioning)
                if cam_id not in self.camera_workers:
                    logging.info(f"Nova câmara detetada: '{cam_id}'. A iniciar Worker...")
                    
                    cam_queue = queue.Queue(maxsize=30)
                    self.camera_queues[cam_id] = cam_queue
                    
                    worker = CameraWorker(
                        cam_id=cam_id,
                        input_queue=cam_queue,
                        config=self.config,
                        global_manager=self.global_id_manager,
                        stop_event=self.stop_event
                    )
                    worker.start()
                    self.camera_workers[cam_id] = worker

                # Envia o frame para a fila do Worker correspondente
                try:
                    self.camera_queues[cam_id].put_nowait(frame)
                except queue.Full:
                    # Se o Worker estiver lento (GPU estrangulada), descartamos o frame 
                    # para evitar latência progressiva (Lag).
                    pass

            except queue.Empty:
                pass
            except KeyboardInterrupt:
                self.stop_event.set()

    def _shutdown(self):
        logging.info("A encerrar o sistema...")
        self.stop_event.set()

        # Aguarda que a extração pare
        if self._reader_invoker:
            self._reader_invoker.join(timeout=3)

        # Exporta as estatísticas
        self.global_id_manager.export_data_to_csv("tracking_data_final.csv")
        cv2.destroyAllWindows()
        logging.info("Programa finalizado.")