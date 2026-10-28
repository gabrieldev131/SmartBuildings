# extraction/CameraProcess.py
import logging
import multiprocessing
import threading
import time

from core.runtime_setup import apply as apply_runtime_setup
from extraction.frameReaderCommand.FrameReaderInvoker import FrameReaderInvoker
from ui.display import MultiCameraDisplay
from vision.cameraWorker import CameraWorker


def run_camera_process(
    command_factory,
    config,
    global_manager_proxy,
    stop_event: multiprocessing.Event,
    proc_name: str,
) -> None:
    """
    Worker entrypoint for legacy multiprocess mode (1 OS process per camera source).
    Uses FrameReaderInvoker for asynchronous I/O, CameraWorker for GPU inference + tracking,
    and MultiCameraDisplay for decoupled GUI presentation.
    """
    apply_runtime_setup()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [{proc_name}/%(threadName)s] - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    local_stop = threading.Event()

    def _bridge_stop():
        stop_event.wait()
        local_stop.set()

    threading.Thread(target=_bridge_stop, daemon=True, name=f"{proc_name}-stop-bridge").start()

    command = command_factory()
    reader = FrameReaderInvoker(
        cam_key=proc_name,
        command=command,
        stop_event=local_stop,
        name=f"{proc_name}-reader",
    )
    reader.start()

    display = MultiCameraDisplay(config)
    from vision.BatchInferenceEngine import BatchInferenceEngine
    engine = BatchInferenceEngine(config)
    workers: dict[str, CameraWorker] = {}

    try:
        while not local_stop.is_set():
            frames_dict, new_cams = reader.get_batch_state()
            if not frames_dict:
                time.sleep(0.001)
                continue

            cam_ids = sorted(frames_dict.keys())
            frames_list = [frames_dict[cid] for cid in cam_ids]
            det_results = engine.infer_batch(frames_list)
            now_ts = time.time()

            rendered_frames = {}
            for idx, cam_id in enumerate(cam_ids):
                is_first = cam_id not in workers
                if is_first:
                    logging.info(f"Nova câmera detectada automaticamente: '{cam_id}'...")
                    workers[cam_id] = CameraWorker(
                        cam_id=cam_id,
                        config=config,
                        global_manager=global_manager_proxy,
                        inference_engine=engine,
                    )

                boxes_xyxy, confs, feats = det_results[idx]
                disp_frame = workers[cam_id].process_detections(
                    frame=frames_list[idx],
                    boxes_xyxy=boxes_xyxy,
                    confs=confs,
                    feats=feats,
                    current_time=now_ts,
                    is_new_frame=(is_first or (cam_id in new_cams)),
                )
                rendered_frames[cam_id] = disp_frame

            if not display.render(rendered_frames):
                stop_event.set()
                local_stop.set()
                break

    except KeyboardInterrupt:
        stop_event.set()
        local_stop.set()

    finally:
        local_stop.set()
        reader.join(timeout=2.0)
        for worker in workers.values():
            worker.cleanup()
        engine.cleanup()
        display.cleanup()
        logging.info(f"[{proc_name}] Processo encerrado.")
