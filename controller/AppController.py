# controller/app_controller.py
import threading
import queue
import time
import logging
from multiprocessing import Pool
from multiprocessing import Queue as MPQueue
from model.ProcessingWorker import initialize_worker
from controller.CameraProcessor import CameraProcessor

# Importações do novo padrão Command
from model.frameReaderCommand.ReadKafkaCommand import ReadKafkaCommand
from model.frameReaderCommand.FrameReaderInvoker import FrameReaderInvoker

from model.GlobalIdentityManager import GlobalIdentityManager

class AppController:
    """
    Orquestra o ciclo de vida da aplicação.

    Mudança de arquitetura (Command Pattern):
      - A captura de frames agora é agnóstica em relação à fonte.
      - O AppController instancia um comando (ReadKafkaCommand, ReadRTSPCommand)
        e passa-o para um Invoker (FrameReaderInvoker) executar numa thread isolada.
    """

    def __init__(self, view, config):
        self._view = view
        self._config = config

        self.camera_processors: dict[str, CameraProcessor] = {}
        self.raw_frames_queue: MPQueue = None
        self.stop_event: threading.Event = None
        self._reader_invoker: FrameReaderInvoker = None
        self.pool: Pool = None

        self.global_id_manager = GlobalIdentityManager(similarity_threshold=0.4)

        # Métricas de performance
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self._current_fps = 0.0

    # -------------------------------------------------------------------------
    # CICLO DE VIDA
    # -------------------------------------------------------------------------

    def run(self):
        if self._setup():
            self._main_loop()
        self._shutdown()

    def _setup(self):
        self._view.display_message("A iniciar o sistema (Padrão Command)...")
        self.stop_event = threading.Event()

        self._start_frame_reader()
        self._initialize_process_pool()

        self._view.display_message(
            f"A aguardar frames de processamento...")
        return True

    def _main_loop(self):
        while not self.stop_event.is_set():
            self._process_one_frame()
            self._update_fps()
            if self._view.get_keypress() == ord("q"):
                self.stop_event.set()

    def _shutdown(self):
        self._view.display_message("A encerrar o sistema...")

        self.global_id_manager.export_data_to_csv("dados_rastreamento.csv")
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()

        if self.pool:
            self.pool.terminate()
            self.pool.join()

        if self.raw_frames_queue:
            self.raw_frames_queue.cancel_join_thread()
            self.raw_frames_queue.close()

        # O Invoker encarrega-se do join e do cleanup do comando
        if self._reader_invoker:
            self._reader_invoker.join(timeout=3)

        self._view.destroy_windows()
        logging.info("Programa finalizado.")

    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    

    def _start_frame_reader(self):
        """
        Cria a fila partilhada e inicia o Padrão Command para leitura de frames.
        """
        self.raw_frames_queue = MPQueue(
            maxsize=60)

        # 1. Instanciamos o Comando (A intenção do que queremos fazer)
        kafka_command = ReadKafkaCommand(
            output_queue=self.raw_frames_queue,
            bootstrap_servers=self._config.KAFKA_BOOTSTRAP_SERVERS,
            topic=self._config.KAFKA_TOPIC,
            group_id=self._config.KAFKA_GROUP_ID,
            width=self._config.PROCESSING_WIDTH,
            height=self._config.PROCESSING_HEIGHT,
            target_camera_id=getattr(self._config, "KAFKA_TARGET_CAMERA", None)
        )

        # 2. Instanciamos o Invoker (Quem gere a thread e executa o comando)
        self._reader_invoker = FrameReaderInvoker(
            command=kafka_command,
            stop_event=self.stop_event,
            name="KafkaReaderInvoker"
        )
        
        # 3. Iniciamos a thread
        self._reader_invoker.start()

    def _initialize_process_pool(self):
        if self._config.USE_MULTIPROCESSING:
            self.pool = Pool(
                processes=self._config.NUM_WORKER_PROCESSES,
                initializer=initialize_worker,
                initargs=[
                    getattr(self._config, "YOLO_MODEL_PATH", "yolov8n.pt")
                ],
            )

    # -------------------------------------------------------------------------
    # LOOP PRINCIPAL
    # -------------------------------------------------------------------------

    def _process_one_frame(self):
        try:
            cam_id, frame = self.raw_frames_queue.get(timeout=0.1)

            if cam_id not in self.camera_processors:
                logging.info(f"Nova câmara detetada: '{cam_id}'")
                self._view.display_message(f"Câmara '{cam_id}' ligada.")
                detection_lock = threading.Lock()
                
                # -------------------------------------------------------------
                # INTEGRAÇÃO: Injetamos o global_id_manager no construtor
                # Assim, todos os CameraProcessors conversam com o mesmo objeto.
                # -------------------------------------------------------------
                self.camera_processors[cam_id] = CameraProcessor(
                    cam_id, 
                    self._config, 
                    detection_lock,
                    global_id_manager=self.global_id_manager # <--- INJEÇÃO AQUI
                )

            processor = self.camera_processors[cam_id]
            processed_frame = processor.process_frame(frame, self.pool)
            self._draw_fps_overlay(processed_frame, cam_id)
            self._view.show_frame(f"Câmara {cam_id}", processed_frame)

        except queue.Empty:
            pass
        except OSError:
            pass

    # -------------------------------------------------------------------------
    # MÉTRICAS E OVERLAY
    # -------------------------------------------------------------------------

    def _update_fps(self):
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_last_time = now

    def _draw_fps_overlay(self, frame, cam_id: str):
        import cv2
        queue_size = self.raw_frames_queue.qsize()
        fps_color = (0, 255, 0) if self._current_fps >= 15 else (0, 165, 255)
        text = f"FPS: {self._current_fps:.1f} | Fila: {queue_size} | {cam_id}"
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 1, cv2.LINE_AA)