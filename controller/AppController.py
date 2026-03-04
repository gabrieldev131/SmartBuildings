# controller/app_controller.py
import threading
import queue
import time
import logging
from multiprocessing import Pool
from multiprocessing import Queue as MPQueue
from model.ProcessingWorker import initialize_worker
from controller.CameraProcessor import CameraProcessor

class AppController:
    def __init__(self, model, view, config):
        self._view = view
        self._config = config
        self._scanner = model['scanner']
        self._frame_reader_class = model['frame_reader_class']
        
        self.camera_processors = {}
        self.raw_frames_queue = None
        self.stop_event = None
        self.readers = []
        self.pool = None

        # --- Metricas de Performance ---
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self._current_fps = 0.0

    def run(self):
        if self._setup():
            self._main_loop()
        self._shutdown()

    def _setup(self):
        self._view.display_message("Iniciando configuracao do sistema...")
        self._initialize_concurrency_primitives()
        
        rtsp_urls = self._discover_cameras()
        if not rtsp_urls:
            self._view.display_message("Nenhuma camera encontrada. Encerrando.")
            return False

        self._start_frame_readers(rtsp_urls)
        self._initialize_process_pool()
        
        self._view.display_message("Configuracao concluida. Iniciando visualizacao...")
        return True

    def _main_loop(self):
        while not self.stop_event.is_set():
            self._process_one_frame()
            self._update_fps()
            if self._view.get_keypress() == ord('q'):
                self.stop_event.set()

    def _shutdown(self):
        self._view.display_message("Iniciando procedimento de encerramento...")
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
        self._shutdown_process_pool()
        self._shutdown_queues_and_threads()
        self._view.destroy_windows()
        logging.info("Programa finalizado com sucesso.")

    def _initialize_concurrency_primitives(self):
        self.stop_event = threading.Event()

    def _discover_cameras(self):
        self._view.display_scan_start(
            f"{self._config.NETWORK_BASE}.x", self._config.RTSP_PORT)
        discovered_ips = self._scanner.discover(
            progress_callback=self._view.display_camera_found)
        self._view.display_scan_complete(len(discovered_ips))
        return [
            f"rtsp://{self._config.CAMERA_USERNAME}:{self._config.CAMERA_PASSWORD}"
            f"@{ip}:{self._config.RTSP_PORT}/Streaming/Channels/{self._config.STREAM_TYPE}"
            for ip in discovered_ips
        ]

    def _start_frame_readers(self, rtsp_urls):
        # CORRECAO: MPQueue e process-safe E thread-safe.
        # A versao anterior usava multiprocessing.Queue no import mas
        # havia ambiguidade com queue.Queue do FrameReader.
        # Agora importamos explicitamente como MPQueue para clareza.
        self.raw_frames_queue = MPQueue(maxsize=len(rtsp_urls) * self._config.BUFFER_SIZE)
        
        for url in rtsp_urls:
            # Cada camera recebe seu proprio lock para proteger pending_detections
            # contra a race condition do callback do pool.apply_async
            detection_lock = threading.Lock()
            self.camera_processors[url] = CameraProcessor(url, self._config, detection_lock)
            
            reader = self._frame_reader_class(
                url, self.raw_frames_queue, self.stop_event,
                self._config.PROCESSING_WIDTH, self._config.PROCESSING_HEIGHT)
            reader.start()
            self.readers.append(reader)

    def _initialize_process_pool(self):
        if self._config.USE_MULTIPROCESSING:
            self.pool = Pool(
                processes=self._config.NUM_WORKER_PROCESSES,
                initializer=initialize_worker,
                initargs=(
                    self._config.YOLO_WEIGHTS,
                    self._config.YOLO_CFG,
                    self._config.YOLO_NAMES
                )
            )

    def _process_one_frame(self):
        try:
            source_id, frame = self.raw_frames_queue.get(timeout=0.1)
            if source_id in self.camera_processors:
                processor = self.camera_processors[source_id]
                processed_frame = processor.process_frame(frame, self.pool)
                self._draw_fps_overlay(processed_frame)
                self._view.show_frame(f"Camera {source_id}", processed_frame)
        except queue.Empty:
            pass
        except OSError:
            # MPQueue lanca OSError quando fechada durante shutdown
            pass

    def _update_fps(self):
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_last_time = now

    def _draw_fps_overlay(self, frame):
        import cv2
        queue_size = self.raw_frames_queue.qsize()
        # FPS: saudavel acima de 15. Fila alta indica gargalo de processamento.
        fps_color = (0, 255, 0) if self._current_fps >= 15 else (0, 165, 255)
        text = f"FPS: {self._current_fps:.1f} | Fila: {queue_size}"
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 1, cv2.LINE_AA)

    def _shutdown_process_pool(self):
        if self.pool:
            self._view.display_message("Encerrando processos trabalhadores...")
            self.pool.terminate()
            self.pool.join()

    def _shutdown_queues_and_threads(self):
        if self.raw_frames_queue:
            self.raw_frames_queue.cancel_join_thread()
            self.raw_frames_queue.close()
        for reader in self.readers:
            reader.thread.join(timeout=2)
        self._view.display_message("Threads de leitura finalizadas.")