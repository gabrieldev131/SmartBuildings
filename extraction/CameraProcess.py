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
    workers: dict[str, CameraWorker] = {}

    try:
        while not local_stop.is_set():
            frames_dict = reader.get_latest()
            if frames_dict is None:
                time.sleep(0.001)
                continue

            rendered_frames = {}
            for cam_id, frame in frames_dict.items():
                if cam_id not in workers:
                    logging.info(f"Nova câmera detectada automaticamente: '{cam_id}'...")
                    workers[cam_id] = CameraWorker(
                        cam_id=cam_id,
                        config=config,
                        global_manager=global_manager_proxy,
                    )

                worker = workers[cam_id]
                disp_frame = worker.process_frame(frame)
                rendered_frames[cam_id] = disp_frame if disp_frame is not None else frame

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
        display.cleanup()
        logging.info(f"[{proc_name}] Processo encerrado.")
