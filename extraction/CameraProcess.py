# extraction/CameraProcess.py
import logging
import multiprocessing
import threading
import time

from identity.runtime_setup import apply as apply_runtime_setup
from extraction.frameReaderCommand.FrameReaderInvoker import FrameReaderInvoker
from ui.display import MultiCameraDisplay
from vision.BatchInferenceEngine import BatchInferenceEngine
from vision.cameraWorker import CameraWorker


def _setup_process_logging(proc_name: str, stop_event: multiprocessing.Event) -> threading.Event:
    apply_runtime_setup()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [{proc_name}/%(threadName)s] - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    local_stop = threading.Event()
    threading.Thread(target=lambda: (stop_event.wait(), local_stop.set()), daemon=True, name=f"{proc_name}-stop").start()
    return local_stop


def _process_camera_step(reader: FrameReaderInvoker, engine: BatchInferenceEngine, workers: dict, config, global_manager_proxy) -> dict:
    frames_dict, new_cams = reader.get_batch_state()
    if not frames_dict:
        time.sleep(0.001)
        return {}
    cam_ids = sorted(frames_dict.keys())
    frames_list = [frames_dict[cid] for cid in cam_ids]
    det_results = engine.infer_batch(frames_list)
    now_ts = time.time()
    rendered = {}
    for idx, cid in enumerate(cam_ids):
        is_first = cid not in workers
        if is_first:
            workers[cid] = CameraWorker(cid, config, global_manager_proxy, engine)
        rendered[cid] = workers[cid].process_detections(frames_list[idx], *det_results[idx], current_time=now_ts, is_new_frame=(is_first or (cid in new_cams)))
    return rendered


def _cleanup_camera_process(local_stop: threading.Event, reader: FrameReaderInvoker, workers: dict, engine: BatchInferenceEngine, display: MultiCameraDisplay, proc_name: str) -> None:
    local_stop.set()
    reader.join(timeout=2.0)
    for worker in workers.values():
        worker.cleanup()
    engine.cleanup()
    display.cleanup()
    logging.info(f"[{proc_name}] Processo encerrado.")


def run_camera_process(command_factory, config, global_manager_proxy, stop_event: multiprocessing.Event, proc_name: str) -> None:
    """Worker entrypoint for legacy multiprocess mode (1 OS process per camera source)."""
    local_stop = _setup_process_logging(proc_name, stop_event)
    reader = FrameReaderInvoker(proc_name, command_factory(), local_stop, name=f"{proc_name}-reader")
    reader.start()
    display, engine, workers = MultiCameraDisplay(config), BatchInferenceEngine(config), {}
    try:
        while not local_stop.is_set():
            rendered = _process_camera_step(reader, engine, workers, config, global_manager_proxy)
            if rendered and not display.render(rendered):
                stop_event.set()
                local_stop.set()
                break
    except KeyboardInterrupt:
        stop_event.set()
        local_stop.set()
    finally:
        _cleanup_camera_process(local_stop, reader, workers, engine, display, proc_name)
