# extraction/FrameReaderManagement.py
"""
Application Orchestrator (Facade Pattern).
Coordinates camera command factories, asynchronous frame readers, the Unified Batched
CUDA-Graph Inference Engine, per-camera VectorizedDeepSort workers, Cross-Camera
GlobalIdentityManager, and the MultiCameraDisplay presenter.
"""
import logging
import multiprocessing as mp
import threading
import time

from core.GlobalIdentityManager import GlobalIdentityManager
from core.IdentityManagerServer import IdentityManagerServer
from extraction.CameraProcess import run_camera_process
from extraction.frameReaderCommand.CommandFactories import RTSPCommandFactory, KafkaCommandFactory
from extraction.frameReaderCommand.FrameReaderInvoker import FrameReaderInvoker
from ui.display import MultiCameraDisplay
from vision.BatchInferenceEngine import BatchInferenceEngine
from vision.cameraWorker import CameraWorker


class FrameReaderManagement:
    """
    Central system orchestrator supporting two execution strategies configurable via Config.EXECUTION_MODE:
    1) "batched" (Default, High Performance):
       - Runs one dedicated I/O FrameReaderInvoker thread per camera source.
       - Runs a single centralized BatchInferenceEngine on GPU with captured CUDA Graphs & FP16.
       - Uses Double-Buffered Software Pipelining (GPU executes Batch t while CPU tracks Batch t-1),
         delivering 60-100+ FPS per camera across up to 8 cameras on an RTX 2080 Ti.
       - Shares GlobalIdentityManager directly in-memory with zero socket/pickle IPC overhead.
    2) "multiprocess" (Legacy Distributed Mode):
       - Spawns one OS process per camera communicating via IdentityManagerServer (BaseManager).
    """

    def __init__(self, config):
        self.config = config
        self.stop_event = mp.Event()
        self.local_stop = threading.Event()
        self.processes: list[mp.Process] = []
        self.readers: list[FrameReaderInvoker] = []
        self.manager = None
        self.global_manager = None

    def _camera_factories(self):
        """
        Configure all active camera sources here.
        Add up to 8 (or more) RTSPCommandFactory or KafkaCommandFactory entries.
        """
        w = self.config.PROCESSING_WIDTH
        h = self.config.PROCESSING_HEIGHT
        loop = getattr(self.config, "LOOP_LOCAL_VIDEOS", True)
        pace = getattr(self.config, "VIDEO_FILE_REALTIME_PACING", False)

        return [
            #RTSPCommandFactory(
            #    source="rtsp://admin:Aluno@00@10.145.80.52:554",
            #    width=w,
            #    height=h,
            #    cam_id="cam_1",
            #    loop_video=loop,
            #    pace_fps=pace,
            #),
            # Example local video or additional RTSP cameras (uncomment to scale up to 8 cameras):
            # RTSPCommandFactory(source="models/pessoas.mp4", width=w, height=h, cam_id="cam_2", loop_video=loop, pace_fps=pace),
            KafkaCommandFactory(
                bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS,
                topic=self.config.KAFKA_TOPIC,
                group_id=self.config.KAFKA_GROUP_ID,
                width=w,
                height=h,
                target_camera_id=self.config.KAFKA_TARGET_CAMERA,
            ),
        ]

    def run(self) -> None:
        mode = getattr(self.config, "EXECUTION_MODE", "batched").lower()
        logging.info(f"A iniciar o sistema em modo '{mode}'...")

        if mode == "multiprocess":
            self._run_multiprocess_mode()
        else:
            self._run_batched_pipeline_mode()

    # =========================================================================
    # MODE 1: HIGH-PERFORMANCE BATCHED GPU + DOUBLE-BUFFERED PIPELINE (60+ FPS)
    # =========================================================================
    def _run_batched_pipeline_mode(self) -> None:
        self.global_manager = GlobalIdentityManager(self.config)
        engine = BatchInferenceEngine(self.config)
        display = MultiCameraDisplay(self.config)
        workers: dict[str, CameraWorker] = {}

        factories = self._camera_factories()
        for idx, factory in enumerate(factories):
            cmd = factory()
            reader_name = f"CameraReader-{idx}"
            invoker = FrameReaderInvoker(
                cam_key=reader_name,
                command=cmd,
                stop_event=self.local_stop,
                name=reader_name,
            )
            invoker.start()
            self.readers.append(invoker)
            logging.info(f"Leitor assíncrono '{reader_name}' iniciado.")

        # Wait briefly for initial frames from active cameras
        self._wait_for_initial_frames(timeout=3.0)

        prev_batch_state = None  # Holds (cam_ids, frames_list, det_results, batch_timestamp)
        step_idx = 0

        def _process_and_render(batch_state) -> bool:
            p_cam_ids, p_frames, p_det_results, p_ts = batch_state
            rendered_dict = {}
            for c_idx, cid in enumerate(p_cam_ids):
                if cid not in workers:
                    logging.info(f"Nova câmera ativa no pipeline: '{cid}'")
                    workers[cid] = CameraWorker(
                        cam_id=cid,
                        config=self.config,
                        global_manager=self.global_manager,
                        inference_engine=engine,
                    )
                boxes_xyxy, confs, feats = p_det_results[c_idx]
                out_frame = workers[cid].process_detections(
                    frame=p_frames[c_idx],
                    boxes_xyxy=boxes_xyxy,
                    confs=confs,
                    feats=feats,
                    current_time=p_ts,
                )
                rendered_dict[cid] = out_frame

            if not display.render(rendered_dict):
                self.local_stop.set()
                return False
            return True

        try:
            while not self.local_stop.is_set():
                cam_ids, frames_list = self._collect_batch_frames()
                if not frames_list:
                    # Flush any staged GPU batch immediately so 30 FPS Kafka/RTSP streams never wait
                    if prev_batch_state is not None:
                        if not _process_and_render(prev_batch_state):
                            break
                        prev_batch_state = None
                    if not display.poll_key():
                        self.local_stop.set()
                        break
                    time.sleep(0.001)
                    continue

                buf_idx = step_idx % 2
                now_wall = time.time()

                # 1. Asynchronously launch GPU inference (YOLO + ROI-Align + Re-ID CUDA Graph) for Batch t
                curr_slot, curr_buf = engine.launch_async(frames_list, buf_idx=buf_idx)

                # 2. While GPU hardware executes Batch t, CPU processes tracking & annotations for Batch t-1
                if prev_batch_state is not None:
                    if not _process_and_render(prev_batch_state):
                        break

                # 3. Synchronize GPU stream for Batch t and stage its results for next iteration's CPU step
                det_results = engine.synchronize_and_filter(curr_slot, curr_buf)
                prev_batch_state = (cam_ids, frames_list, det_results, now_wall)
                step_idx += 1

        except KeyboardInterrupt:
            logging.info("Interrupção manual (Ctrl+C). Encerrando pipeline em lote...")
        finally:
            self.local_stop.set()
            for r in self.readers:
                r.join(timeout=2.0)
            for w in workers.values():
                w.cleanup()
            engine.cleanup()
            display.cleanup()
            logging.info("A exportar dados para CSV...")
            self.global_manager.export_data_to_csv("tracking_data_final.csv")
            logging.info("Programa finalizado com sucesso.")

    def _wait_for_initial_frames(self, timeout: float = 3.0) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            ready = sum(1 for r in self.readers if r._cached_frames)
            if ready == len(self.readers) or ready > 0:
                break
            time.sleep(0.02)

    def _collect_batch_frames(self) -> tuple[list[str], list]:
        cam_ids = []
        frames_list = []

        for r in self.readers:
            frames_dict = r.get_latest()
            if not frames_dict:
                continue
            for cid, frm in frames_dict.items():
                cam_ids.append(cid)
                frames_list.append(frm)

        return cam_ids, frames_list

    # =========================================================================
    # MODE 2: LEGACY MULTIPROCESS MODE (1 OS Process Per Camera)
    # =========================================================================
    def _run_multiprocess_mode(self) -> None:
        self._start_identity_server()
        self._start_camera_processes()
        self._wait_for_processes()
        self._shutdown_multiprocess()

    def _start_identity_server(self) -> None:
        self.manager = IdentityManagerServer(authkey=b"smartbuilds-reid")
        self.manager.start()
        self.global_manager = self.manager.GlobalIdentityManager(self.config)
        logging.info("Servidor de identidades globais iniciado.")

    def _start_camera_processes(self) -> None:
        for idx, factory in enumerate(self._camera_factories()):
            proc_name = f"CameraProc-{idx}"
            p = mp.Process(
                target=run_camera_process,
                args=(factory, self.config, self.global_manager, self.stop_event, proc_name),
                name=proc_name,
            )
            p.start()
            self.processes.append(p)
            logging.info(f"Processo '{proc_name}' iniciado (pid={p.pid}).")

    def _wait_for_processes(self) -> None:
        try:
            while any(p.is_alive() for p in self.processes):
                for p in self.processes:
                    p.join(timeout=0.5)
        except KeyboardInterrupt:
            logging.info("Interrupção manual (Ctrl+C). Encerrando todos os processos...")

    def _shutdown_multiprocess(self) -> None:
        logging.info("A iniciar rotina de encerramento seguro...")
        self.stop_event.set()

        for p in self.processes:
            p.join(timeout=5.0)
            if p.is_alive():
                logging.warning(f"Processo '{p.name}' não encerrou a tempo, finalizando à força.")
                p.terminate()

        logging.info("A exportar dados para CSV...")
        self.global_manager.export_data_to_csv("tracking_data_final.csv")
        self.manager.shutdown()
        logging.info("Programa finalizado com sucesso.")
