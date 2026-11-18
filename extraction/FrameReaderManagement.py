# extraction/FrameReaderManagement.py
import logging
import multiprocessing as mp
import threading
import time

from identity.GlobalIdentityManager import GlobalIdentityManager
from identity.IdentityManagerServer import IdentityManagerServer
from extraction.CameraProcess import run_camera_process
from extraction.frameReaderCommand.CommandFactories import KafkaCommandFactory, RTSPCommandFactory
from extraction.frameReaderCommand.FrameReaderInvoker import FrameReaderInvoker
from ui.display import MultiCameraDisplay
from vision.BatchInferenceEngine import BatchInferenceEngine
from vision.cameraWorker import CameraWorker


class FrameReaderManagement:
    """Central orchestrator supporting 'batched' (default) and 'multiprocess' execution modes (SRP & DIP)."""

    def __init__(self, config):
        self.config = config
        self.stop_event = mp.Event()
        self.local_stop = threading.Event()
        self.processes: list[mp.Process] = []
        self.readers: list[FrameReaderInvoker] = []
        self.manager = None
        self.global_manager = None

    def _camera_factories(self) -> list:
        w, h = self.config.PROCESSING_WIDTH, self.config.PROCESSING_HEIGHT
        kafka_bootstrap = getattr(self.config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        return [
            # RTSPCommandFactory(source="testes/pessoas.mp4", width=w, height=h, cam_id="cam_1"),
            KafkaCommandFactory(
                bootstrap_servers=kafka_bootstrap, topic=self.config.KAFKA_TOPIC,
                group_id=self.config.KAFKA_GROUP_ID, width=w, height=h,
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

    def _start_readers(self) -> None:
        for idx, factory in enumerate(self._camera_factories()):
            name = f"CameraReader-{idx}"
            invoker = FrameReaderInvoker(cam_key=name, command=factory(), stop_event=self.local_stop, name=name)
            invoker.start()
            self.readers.append(invoker)
            logging.info(f"Leitor assíncrono '{name}' iniciado.")
        self._wait_for_initial_frames(timeout=3.0)

    def _process_and_render_batch(self, batch_state: tuple, workers: dict, engine: BatchInferenceEngine, display: MultiCameraDisplay) -> bool:
        p_cam_ids, p_frames, p_det_results, p_ts, p_new_cams = batch_state
        rendered = {}
        for c_idx, cid in enumerate(p_cam_ids):
            is_first = cid not in workers
            if is_first:
                logging.info(f"Nova câmera ativa no pipeline: '{cid}'")
                workers[cid] = CameraWorker(cid, self.config, self.global_manager, engine)
            rendered[cid] = workers[cid].process_detections(
                p_frames[c_idx], *p_det_results[c_idx], current_time=p_ts, is_new_frame=(is_first or (cid in p_new_cams))
            )
        if not display.render(rendered):
            self.local_stop.set()
            return False
        return True

    def _handle_empty_batch(self, prev_state, workers: dict, engine: BatchInferenceEngine, display: MultiCameraDisplay) -> bool:
        if prev_state is not None and not self._process_and_render_batch(prev_state, workers, engine, display):
            return False
        if not display.poll_key():
            self.local_stop.set()
            return False
        time.sleep(0.001)
        return True

    def _run_batched_loop(self, engine: BatchInferenceEngine, display: MultiCameraDisplay, workers: dict) -> None:
        prev_state, step_idx = None, 0
        while not self.local_stop.is_set():
            cam_ids, frames_list, new_cams = self._collect_batch_frames()
            if not frames_list:
                if not self._handle_empty_batch(prev_state, workers, engine, display):
                    break
                prev_state = None
                continue
            slot, buf = engine.launch_async(frames_list, buf_idx=step_idx % 2)
            if prev_state is not None and not self._process_and_render_batch(prev_state, workers, engine, display):
                break
            prev_state = (cam_ids, frames_list, engine.synchronize_and_filter(slot, buf), time.time(), new_cams)
            step_idx += 1

    def _shutdown_batched_pipeline(self, workers: dict, engine: BatchInferenceEngine, display: MultiCameraDisplay) -> None:
        self.local_stop.set()
        for r in self.readers:
            r.join(timeout=2.0)
        for w in workers.values():
            w.cleanup()
        engine.cleanup()
        display.cleanup()
        logging.info("A exportar dados consolidados...")
        self.global_manager.export_data()
        logging.info("Programa finalizado com sucesso.")

    def _run_batched_pipeline_mode(self) -> None:
        self.global_manager = GlobalIdentityManager(self.config)
        engine = BatchInferenceEngine(self.config)
        display = MultiCameraDisplay(self.config)
        workers: dict[str, CameraWorker] = {}
        self._start_readers()
        try:
            self._run_batched_loop(engine, display, workers)
        except KeyboardInterrupt:
            logging.info("Interrupção manual (Ctrl+C). Encerrando pipeline em lote...")
        finally:
            self._shutdown_batched_pipeline(workers, engine, display)

    def _wait_for_initial_frames(self, timeout: float = 3.0) -> None:
        deadline = time.perf_counter() + timeout
        expected = max(len(self.readers), getattr(self.config, "KAFKA_EXPECTED_CAMERAS", 2))
        first_ts = None
        while time.perf_counter() < deadline:
            total = sum(len(r._cached_frames) for r in self.readers)
            if total >= expected or (first_ts is not None and (time.perf_counter() - first_ts) >= 0.40):
                break
            if total > 0 and first_ts is None:
                first_ts = time.perf_counter()
            time.sleep(0.01)

    def _collect_batch_frames(self) -> tuple[list[str], list, set[str]]:
        frames_by_cam, new_cams = {}, set()
        for r in self.readers:
            cached_dict, r_new_cams = r.get_batch_state()
            if cached_dict:
                frames_by_cam.update(cached_dict)
                new_cams.update(r_new_cams)
        if not frames_by_cam:
            return [], [], set()
        cam_ids = sorted(frames_by_cam.keys())
        return cam_ids, [frames_by_cam[cid] for cid in cam_ids], new_cams

    def _spawn_camera_processes(self) -> None:
        for idx, factory in enumerate(self._camera_factories()):
            name = f"CameraProc-{idx}"
            p = mp.Process(target=run_camera_process, args=(factory, self.config, self.global_manager, self.stop_event, name), name=name)
            p.start()
            self.processes.append(p)
            logging.info(f"Processo '{name}' iniciado (pid={p.pid}).")

    def _shutdown_multiprocess(self) -> None:
        self.stop_event.set()
        for p in self.processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        self.global_manager.export_data()
        self.manager.shutdown()
        logging.info("Programa finalizado com sucesso.")

    def _run_multiprocess_mode(self) -> None:
        self.manager = IdentityManagerServer(authkey=b"smartbuilds-reid")
        self.manager.start()
        self.global_manager = self.manager.GlobalIdentityManager(self.config)
        self._spawn_camera_processes()
        try:
            while any(p.is_alive() for p in self.processes):
                for p in self.processes:
                    p.join(timeout=0.5)
        except KeyboardInterrupt:
            logging.info("Interrupção manual (Ctrl+C). Encerrando processos...")
        finally:
            self._shutdown_multiprocess()
