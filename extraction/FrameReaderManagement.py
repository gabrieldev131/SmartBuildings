# extraction/FrameReaderManagement.py
import threading
import time
import logging
import cv2

from extraction.frameReaderCommand.ReadRTSPCommand import ReadRTSPCommand
from extraction.frameReaderCommand.ReadKafkaCommand import ReadKafkaCommand

from core.GlobalIdentityManager import GlobalIdentityManager
from vision.cameraWorker import CameraWorker

class FrameReaderManagement:
    """
    Orquestrador único e sequencial para aquisição e processamento de frames.
    """
    def __init__(self, config):
        self.config = config
        self.stop_event = threading.Event()
        
        # Dicionário de workers por câmara (sem filas atreladas)
        self.camera_workers = {}
        
        # Manager de Identidades Único
        self.global_id_manager = GlobalIdentityManager(config)
        self.video_command = None

    def run(self):
        logging.info("A iniciar o sistema...")
        self._start_frame_reader()
        self._main_routing_loop()
        self._shutdown()

    def _start_frame_reader(self):
        """Inicializa a abstração de extração de frames (Command Pattern)."""
        self.video_command = ReadRTSPCommand(
            source="rtsp://admin:Aluno@00@10.145.80.52:554",
            width=640,
            height=480
        )
        
        # Exemplo Kafka (Comente o RTSP acima e descomente abaixo se for usar):
        """
        _target_camera = getattr(self.config, "KAFKA_TARGET_CAMERA", "")
        self.video_command = ReadKafkaCommand(
            bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS,
            topic=self.config.KAFKA_TOPIC,
            group_id=self.config.KAFKA_GROUP_ID,
            width=self.config.PROCESSING_WIDTH,
            height=self.config.PROCESSING_HEIGHT,
            target_camera_id=_target_camera
        )
        """

    def _main_routing_loop(self):
        logging.info("Router principal ativo. Extração e processamento síncronos iniciados.")

        while not self.stop_event.is_set():
            try:
                if self.video_command is None:
                    logging.warning("Leitor de vídeo não inicializado. A tentar inicializar...")
                    self._start_frame_reader()

                if self.video_command is None:
                    time.sleep(0.1)
                    continue

                # 1. Extrai UM frame da fonte
                result = self.video_command.execute()

                if result:
                    cam_id, frame = result
                    
                    # 2. Provisionamento Dinâmico (Instanciação Síncrona do Worker)
                    if cam_id not in self.camera_workers:
                        logging.info(f"Nova câmara detetada: '{cam_id}'. Inicializando Pipeline de Visão...")
                        
                        # --- CORREÇÃO LINUX: FORÇAR CRIAÇÃO DA JANELA ---
                        window_name = f"SmartBuilds - {cam_id}"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, self.config.PROCESSING_WIDTH, self.config.PROCESSING_HEIGHT)
                        # ------------------------------------------------
                        
                        worker = CameraWorker(
                            cam_id=cam_id,
                            config=self.config,
                            global_manager=self.global_id_manager
                        )
                        self.camera_workers[cam_id] = worker

                    # 3. Executa o processamento do frame de imediato (YOLO/Tracking)
                    # NOTA: Assumindo que você criará um método "process_frame" no CameraWorker
                    disp_frame = self.camera_workers[cam_id].process_frame(frame)
                    
                    # 4. Exibe o frame processado ou original na tela
                    if disp_frame is not None:
                        cv2.imshow(f"SmartBuilds - {cam_id}", disp_frame)
                    else:
                        cv2.imshow(f"SmartBuilds - {cam_id}", frame)

                # 5. Processa eventos da interface gráfica
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Tecla 'q' pressionada. Encerrando...")
                    self.stop_event.set()
                    break

            except KeyboardInterrupt:
                logging.info("Interrupção manual detetada (Ctrl+C). Iniciando encerramento...")
                self.stop_event.set()
                break

    def _shutdown(self):
        logging.info("A iniciar rotina de encerramento seguro...")
        self.stop_event.set()

        # 1. Limpa o Command de captura
        if self.video_command:
            logging.info("A libertar recursos de vídeo/rede...")
            self.video_command.cleanup()

        # 2. Limpa os pipelines de visão se necessário
        for cam_id, worker in self.camera_workers.items():
            if hasattr(worker, 'cleanup'):
                worker.cleanup()

        # 3. Exporta as estatísticas
        logging.info("A exportar dados para CSV...")
        self.global_id_manager.export_data_to_csv("tracking_data_final.csv")
        
        # 4. Destrói as janelas com segurança
        cv2.destroyAllWindows()
            
        logging.info("Programa finalizado com sucesso.")