# controller/app_controller.py
import threading
from multiprocessing import Pool, Queue
import queue
from model.processing_worker import initialize_worker
from controller.camera_processor import CameraProcessor

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

    # --- MÉTODO PÚBLICO PRINCIPAL ---
    def run(self):
        """Orquestra o ciclo de vida completo da aplicação."""
        if self._setup():
            self._main_loop()
        self._shutdown()

    # --- MÉTODOS DE CICLO DE VIDA ---
    def _setup(self):
        """Executa toda a sequência de inicialização."""
        self._view.display_message("Iniciando configuração do sistema...")
        self._initialize_concurrency_primitives()
        
        rtsp_urls = self._discover_cameras()
        if not rtsp_urls:
            self._view.display_message("Nenhuma câmera encontrada. Encerrando.")
            return False

        self._start_frame_readers(rtsp_urls)
        self._initialize_process_pool()
        
        self._view.display_message("Configuração concluída. Iniciando visualização...")
        return True

    def _main_loop(self):
        """Contém o loop principal que processa frames até o encerramento."""
        while not self.stop_event.is_set():
            self._process_one_frame()
            if self._view.get_keypress() == ord('q'):
                self.stop_event.set()

    def _shutdown(self):
        """Executa a sequência de limpeza de todos os recursos."""
        self._view.display_message("Iniciando procedimento de encerramento...")
        
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
        
        self._shutdown_process_pool()
        self._shutdown_queues_and_threads()
        self._view.destroy_windows()
        
        print("Programa finalizado com sucesso.")

    # --- MÉTODOS AUXILIARES DE SETUP ---
    def _initialize_concurrency_primitives(self):
        """Cria as filas e eventos de sincronização."""
        self.stop_event = threading.Event()

    def _discover_cameras(self):
        """Usa o scanner para encontrar os URLs RTSP das câmeras."""
        self._view.display_scan_start(
            f"{self._config.NETWORK_BASE}.x", self._config.RTSP_PORT)
        discovered_ips = self._scanner.discover(progress_callback=self._view.display_camera_found)
        self._view.display_scan_complete(len(discovered_ips))
        
        return [f"rtsp://{self._config.CAMERA_USERNAME}:{self._config.CAMERA_PASSWORD}@{ip}:{self._config.RTSP_PORT}/Streaming/Channels/{self._config.STREAM_TYPE}" for ip in discovered_ips]

    def _start_frame_readers(self, rtsp_urls):
        """Cria os processadores e inicia as threads de leitura para cada câmera."""
        self.raw_frames_queue = Queue(maxsize=len(rtsp_urls) * 5)
        for url in rtsp_urls:
            self.camera_processors[url] = CameraProcessor(url, self._config)
            reader = self._frame_reader_class(
                url, self.raw_frames_queue, self.stop_event, 
                self._config.PROCESSING_WIDTH, self._config.PROCESSING_HEIGHT)
            reader.start()
            self.readers.append(reader)

    def _initialize_process_pool(self):
        """Inicializa o pool de processos para a detecção."""
        if self._config.USE_MULTIPROCESSING:
            self.pool = Pool(processes=self._config.NUM_WORKER_PROCESSES, 
                             initializer=initialize_worker,
                             initargs=(self._config.YOLO_WEIGHTS, self._config.YOLO_CFG, self._config.YOLO_NAMES))

    # --- MÉTODOS AUXILIARES DO LOOP PRINCIPAL ---
    def _process_one_frame(self):
        """Tenta pegar e processar um único frame da fila."""
        try:
            source_id, frame = self.raw_frames_queue.get(timeout=0.1) # Timeout curto
            
            if source_id in self.camera_processors:
                processor = self.camera_processors[source_id]
                processed_frame = processor.process_frame(frame, self.pool)
                self._view.show_frame(f"Camera {source_id}", processed_frame)
        except queue.Empty:
            # É normal a fila estar vazia às vezes, apenas continuamos.
            pass          

    # --- MÉTODOS AUXILIARES DE ENCERRAMENTO ---
    def _shutdown_process_pool(self):
        """Encerra o pool de processos."""
        if self.pool:
            self._view.display_message("Encerrando processos trabalhadores...")
            self.pool.terminate()
            self.pool.join()

    def _shutdown_queues_and_threads(self):
        """Limpa a fila e espera as threads finalizarem."""
        if self.raw_frames_queue:
            self._view.display_message("Cancelando espera da fila de frames...")
            self.raw_frames_queue.cancel_join_thread()
            self.raw_frames_queue.close()

        self._view.display_message("Aguardando threads de leitura finalizarem...")
        for reader in self.readers:
            reader.thread.join(timeout=2)
        self._view.display_message("Threads de leitura finalizadas.")