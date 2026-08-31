# extraction/CameraProcess.py
import time
import logging
import threading
import multiprocessing
import cv2

from core.runtime_setup import apply as apply_runtime_setup
from extraction.frameReaderCommand.FrameReaderInvoker import FrameReaderInvoker
from vision.cameraWorker import CameraWorker


def run_camera_process(command_factory, config, global_manager_proxy, stop_event: multiprocessing.Event, proc_name: str):
    """
    Executado inteiramente DENTRO de um processo dedicado a uma câmera
    (chamado via multiprocessing.Process(target=run_camera_process, ...)).

    command_factory: função SEM argumentos que cria o IFrameCommand desta
    câmara (ex.: `lambda: ReadRTSPCommand(...)`). Precisa ser uma factory,
    e não a instância já pronta, por duas razões:
      1) objetos com sockets/handles de vídeo abertos (cv2.VideoCapture,
         consumidor Kafka) geralmente não sobrevivem a ser transferidos
         para outro processo;
      2) com CUDA, o comando é inofensivo, mas o hábito de só construir
         recursos de I/O DEPOIS de já estar no processo definitivo evita
         uma classe inteira de bugs de processo/fork.

    global_manager_proxy: proxy do GlobalIdentityManager (ver
    core/IdentityManagerServer.py). As chamadas de método daqui viajam até
    o processo que guarda o estado de verdade -- é isso que garante que
    esta câmara e todas as outras concordam sobre quem é quem.
    """
    # Cada processo tem seu próprio interpretador: as threads nativas de
    # torch/cv2/BLAS precisam ser limitadas de novo, aqui dentro.
    apply_runtime_setup()

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [{proc_name}/%(threadName)s] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Ponte: o stop_event é do multiprocessing (compartilhado entre
    # processos); o FrameReaderInvoker espera um threading.Event (local a
    # este processo). Uma thread fininha replica um sinal no outro.
    local_stop = threading.Event()

    def _bridge_stop():
        stop_event.wait()
        local_stop.set()

    threading.Thread(target=_bridge_stop, daemon=True, name=f"{proc_name}-stop-bridge").start()

    command = command_factory()
    # Dentro do processo, ainda vale a pena ter uma thread só de captura:
    # aqui não estamos mais tentando paralelizar CPU-bound entre câmaras
    # (isso agora é o processo que resolve), só sobrepor a ESPERA de I/O
    # (rede/decodificação) desta câmera com o processamento dela mesma --
    # e I/O libera o GIL, então uma thread local ainda ajuda.
    reader = FrameReaderInvoker(cam_key=proc_name, command=command, stop_event=local_stop, name=f"{proc_name}-reader")
    reader.start()

    worker = None
    window_name = None

    try:
        while not local_stop.is_set():
            result = reader.get_latest()
            if result is None:
                time.sleep(0.001)
                continue

            cam_id, frame = result

            if worker is None:
                logging.info(f"Inicializando pipeline de visão para '{cam_id}'...")
                window_name = f"SmartBuilds - {cam_id}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
                # O worker recebe o PROXY -- ele chama os mesmos métodos de
                # sempre (get_or_create_global_id, update_existing_identity),
                # sem saber que agora eles viajam até outro processo.
                worker = CameraWorker(cam_id=cam_id, config=config, global_manager=global_manager_proxy)

            disp_frame = worker.process_frame(frame)
            cv2.imshow(window_name, disp_frame if disp_frame is not None else frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                local_stop.set()
                break

    except KeyboardInterrupt:
        stop_event.set()
        local_stop.set()

    finally:
        local_stop.set()
        reader.join(timeout=2.0)
        if worker is not None:
            worker.cleanup()
        if window_name is not None:
            cv2.destroyWindow(window_name)
        logging.info(f"[{proc_name}] Processo encerrado.")