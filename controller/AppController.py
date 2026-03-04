# controller/app_controller.py
import threading
import queue
import time
import logging
from multiprocessing import Pool
from multiprocessing import Queue as MPQueue
from model.ProcessingWorker import initialize_worker
from model.KafkaFrameReader import KafkaFrameReader
from controller.CameraProcessor import CameraProcessor


class AppController:
    """
    Orquestra o ciclo de vida da aplicacao.

    Mudanca de arquitetura (Kafka):
      - NetworkScanner e FrameReader (RTSP) foram removidos do fluxo principal.
      - Um unico KafkaFrameReader substitui todos os FrameReaders individuais:
        ele consome o topico Kafka, descobre os cam_ids pelas chaves das mensagens
        e coloca (cam_id, frame) na MPQueue — mesma interface de antes.
      - CameraProcessors sao criados dinamicamente no _main_loop quando um
        cam_id novo aparece na fila, sem necessidade de pre-descoberta de cameras.
    """

    def __init__(self, model, view, config):
        self._view = view
        self._config = config

        self.camera_processors: dict[str, CameraProcessor] = {}
        self.raw_frames_queue: MPQueue = None
        self.stop_event: threading.Event = None
        self._kafka_reader: KafkaFrameReader = None
        self.pool: Pool = None

        # Metricas de performance
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
        self._view.display_message("Iniciando sistema (fonte: Kafka)...")
        self.stop_event = threading.Event()

        self._start_kafka_reader()
        self._initialize_process_pool()

        self._view.display_message(
            f"Aguardando frames do topico '{self._config.KAFKA_TOPIC}'...")
        return True

    def _main_loop(self):
        while not self.stop_event.is_set():
            self._process_one_frame()
            self._update_fps()
            if self._view.get_keypress() == ord("q"):
                self.stop_event.set()

    def _shutdown(self):
        self._view.display_message("Encerrando sistema...")
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()

        if self.pool:
            self.pool.terminate()
            self.pool.join()

        if self.raw_frames_queue:
            self.raw_frames_queue.cancel_join_thread()
            self.raw_frames_queue.close()

        if self._kafka_reader:
            self._kafka_reader.thread.join(timeout=3)

        self._view.destroy_windows()
        logging.info("Programa finalizado.")

    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------

    def _start_kafka_reader(self):
        """
        Cria a fila compartilhada e inicia o KafkaFrameReader.
        O tamanho da fila usa BUFFER_SIZE do config como capacidade por camera
        estimada (sem saber quantas cameras existem antes de receber msgs).
        """
        estimated_cameras = getattr(self._config, "KAFKA_EXPECTED_CAMERAS", 4)
        self.raw_frames_queue = MPQueue(
            maxsize=estimated_cameras * self._config.BUFFER_SIZE)

        self._kafka_reader = KafkaFrameReader(
            output_queue=self.raw_frames_queue,
            stop_event=self.stop_event,
            bootstrap_servers=self._config.KAFKA_BOOTSTRAP_SERVERS,
            topic=self._config.KAFKA_TOPIC,
            group_id=self._config.KAFKA_GROUP_ID,
            width=self._config.PROCESSING_WIDTH,
            height=self._config.PROCESSING_HEIGHT,
            target_camera_id=getattr(self._config, "KAFKA_TARGET_CAMERA", None),
        )
        self._kafka_reader.start()

    def _initialize_process_pool(self):
        if self._config.USE_MULTIPROCESSING:
            self.pool = Pool(
                processes=self._config.NUM_WORKER_PROCESSES,
                initializer=initialize_worker,
                initargs=(
                    self._config.YOLO_WEIGHTS,
                    self._config.YOLO_CFG,
                    self._config.YOLO_NAMES,
                ),
            )

    # -------------------------------------------------------------------------
    # LOOP PRINCIPAL
    # -------------------------------------------------------------------------

    def _process_one_frame(self):
        """
        Retira um frame da fila e o processa.

        CameraProcessors sao criados sob demanda: quando um cam_id novo
        aparece pela primeira vez, o processor e instanciado automaticamente.
        Isso elimina a necessidade do NetworkScanner para pre-descoberta.
        """
        try:
            cam_id, frame = self.raw_frames_queue.get(timeout=0.1)

            # Criacao dinamica de processor para cameras novas
            if cam_id not in self.camera_processors:
                logging.info(f"Nova camera detectada no Kafka: '{cam_id}'")
                self._view.display_message(f"Camera '{cam_id}' conectada via Kafka.")
                detection_lock = threading.Lock()
                self.camera_processors[cam_id] = CameraProcessor(
                    cam_id, self._config, detection_lock)

            processor = self.camera_processors[cam_id]
            processed_frame = processor.process_frame(frame, self.pool)
            self._draw_fps_overlay(processed_frame, cam_id)
            self._view.show_frame(f"Camera {cam_id}", processed_frame)

        except queue.Empty:
            pass
        except OSError:
            pass

    # -------------------------------------------------------------------------
    # METRICAS E OVERLAY
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