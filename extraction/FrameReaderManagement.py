# extraction/FrameReaderManagement.py
"""
Application Orchestrator.
Coordinates asynchronous camera readers (`FrameReaderInvoker`), double-buffered batched
GPU inference (`BatchInferenceEngine`), per-camera `deep-sort-realtime` trackers (`CameraWorker`),
cross-camera `GlobalIdentityManager`, and OpenCV display (`MultiCameraDisplay`).
"""
import logging
import multiprocessing as mp
import threading
import time

from core.GlobalIdentityManager import GlobalIdentityManager
from core.IdentityManagerServer import IdentityManagerServer
from extraction.CameraProcess import run_camera_process
from extraction.frameReaderCommand.CommandFactories import KafkaCommandFactory, RTSPCommandFactory
from extraction.frameReaderCommand.FrameReaderInvoker import FrameReaderInvoker
from ui.display import MultiCameraDisplay
from vision.BatchInferenceEngine import BatchInferenceEngine
from vision.cameraWorker import CameraWorker


class FrameReaderManagement:
    """Central orchestrator supporting 'batched' (default) and 'multiprocess' execution modes."""

    def __init__(self, config):
        self.config = config
        self.stop_event = mp.Event()
        self.local_stop = threading.Event()
        self.processes: list[mp.Process] = []
        self.readers: list[FrameReaderInvoker] = []
        self.manager = None
        self.global_manager = None

    def _camera_factories(self):
        """Configure active video sources (RTSP, local video files, or Kafka)."""
        w = self.config.PROCESSING_WIDTH
        h = self.config.PROCESSING_HEIGHT
        kafka_bootstrap = getattr(self.config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        return [
            # RTSPCommandFactory(source="testes/pessoas.mp4", width=w, height=h, cam_id="cam_1"),
            KafkaCommandFactory(
                bootstrap_servers=kafka_bootstrap,
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

    def _run_batched_pipeline_mode(self) -> None:
        self.global_manager = GlobalIdentityManager(self.config)
        engine = BatchInferenceEngine(self.config)
        display = MultiCameraDisplay(self.config)
        workers: dict[str, CameraWorker] = {}

        for idx, factory in enumerate(self._camera_factories()):
            reader_name = f"CameraReader-{idx}"
            invoker = FrameReaderInvoker(
                cam_key=reader_name,
                command=factory(),
                stop_event=self.local_stop,
                name=reader_name,
            )
            invoker.start()
            self.readers.append(invoker)
            logging.info(f"Leitor assíncrono '{reader_name}' iniciado.")

        self._wait_for_initial_frames(timeout=3.0)

        prev_batch_state = None
        step_idx = 0

        def _process_and_render(batch_state) -> bool:
            p_cam_ids, p_frames, p_det_results, p_ts, p_new_cams = batch_state
            rendered_dict = {}
            for c_idx, cid in enumerate(p_cam_ids):
                is_first = cid not in workers
                if is_first:
                    logging.info(f"Nova câmera ativa no pipeline: '{cid}'")
                    workers[cid] = CameraWorker(
                        cam_id=cid,
                        config=self.config,
                        global_manager=self.global_manager,
                        inference_engine=engine,
                    )
                boxes_xyxy, confs, feats = p_det_results[c_idx]
                rendered_dict[cid] = workers[cid].process_detections(
                    frame=p_frames[c_idx],
                    boxes_xyxy=boxes_xyxy,
                    confs=confs,
                    feats=feats,
                    current_time=p_ts,
                    is_new_frame=(is_first or (cid in p_new_cams)),
                )

            if not display.render(rendered_dict):
                self.local_stop.set()
                return False
            return True

        try:
            while not self.local_stop.is_set():
                cam_ids, frames_list, new_cams = self._collect_batch_frames()
                if not frames_list:
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

                # 1. Launch GPU inference for Batch t (CUDA Graph / async stream)
                curr_slot, curr_buf = engine.launch_async(frames_list, buf_idx=buf_idx)

                # 2. Overlap CPU DeepSort tracking & rendering for Batch t-1
                if prev_batch_state is not None:
                    if not _process_and_render(prev_batch_state):
                        break

                # 3. Synchronize GPU stream for Batch t
                det_results = engine.synchronize_and_filter(curr_slot, curr_buf)
                prev_batch_state = (cam_ids, frames_list, det_results, now_wall, new_cams)
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
        expected = max(len(self.readers), getattr(self.config, "KAFKA_EXPECTED_CAMERAS", 2))
        first_ready_ts = None
        while time.perf_counter() < deadline:
            total_cams = sum(len(r._cached_frames) for r in self.readers)
            if total_cams >= expected:
                break
            if total_cams > 0:
                if first_ready_ts is None:
                    first_ready_ts = time.perf_counter()
                elif (time.perf_counter() - first_ready_ts) >= 0.40:
                    break
            time.sleep(0.01)

    def _collect_batch_frames(self) -> tuple[list[str], list, set[str]]:
        frames_by_cam = {}
        new_cams: set[str] = set()

        for r in self.readers:
            cached_dict, r_new_cams = r.get_batch_state()
            if cached_dict:
                frames_by_cam.update(cached_dict)
                new_cams.update(r_new_cams)

        if not frames_by_cam:
            return [], [], set()

        cam_ids = sorted(frames_by_cam.keys())
        frames_list = [frames_by_cam[cid] for cid in cam_ids]
        return cam_ids, frames_list, new_cams

    def _run_multiprocess_mode(self) -> None:
        self.manager = IdentityManagerServer(authkey=b"smartbuilds-reid")
        self.manager.start()
        self.global_manager = self.manager.GlobalIdentityManager(self.config)
        logging.info("Servidor de identidades globais iniciado.")

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

        try:
            while any(p.is_alive() for p in self.processes):
                for p in self.processes:
                    p.join(timeout=0.5)
        except KeyboardInterrupt:
            logging.info("Interrupção manual (Ctrl+C). Encerrando processos...")
        finally:
            self.stop_event.set()
            for p in self.processes:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
            self.global_manager.export_data_to_csv("tracking_data_final.csv")
            self.manager.shutdown()
            logging.info("Programa finalizado com sucesso.")
