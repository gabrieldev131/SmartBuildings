# extraction/FrameReaderManagement.py
import multiprocessing as mp
import logging

from core.GlobalIdentityManager import GlobalIdentityManager
from core.IdentityManagerServer import IdentityManagerServer
from extraction.frameReaderCommand import ReadKafkaCommand
from extraction.frameReaderCommand.CommandFactories import RTSPCommandFactory, KafkaCommandFactory

from extraction.CameraProcess import run_camera_process


class FrameReaderManagement:
    """
    Orquestrador único do sistema: sobe o processo servidor de identidades
    globais (IdentityManagerServer) e um processo dedicado por câmara
    (CameraProcess.run_camera_process), aguarda o encerramento e cuida do
    shutdown (export do CSV, etc).

    Este é o ÚNICO lugar que precisa mudar para adicionar, remover ou
    reconfigurar uma câmara -- ver _camera_factories(). main.py fica
    responsável só pelo que é específico do PROCESSO PRINCIPAL (limites de
    threads nativas, método de criação de processos) e chama .run() aqui,
    igual fazia antes com a versão baseada em threads.
    """
    def __init__(self, config):
        self.config = config
        self.stop_event = mp.Event()
        self.processes: list[mp.Process] = []
        self.manager = None
        self.global_manager_proxy = None

    def _camera_factories(self):
        """
        Uma factory por câmara. Para adicionar mais câmaras, inclua mais
        entradas aqui -- cada uma ganha o seu próprio processo, mas todas
        recebem o MESMO global_manager_proxy, então uma pessoa vista em
        câmaras diferentes recebe o mesmo global_id.

        IMPORTANTE: cada entrada precisa ser uma instância de factory
        picklable (classe com __call__, ver CommandFactories.py), NUNCA
        uma lambda -- multiprocessing com "spawn" precisa serializar tudo
        que é passado para Process(...).
        """
        _target_camera = self.config.KAFKA_TARGET_CAMERA
        return [
            #RTSPCommandFactory(source="models/pessoas.mp4", width=640, height=480),
            # Segunda câmara real, por exemplo:
            #RTSPCommandFactory(source="rtsp://admin:Aluno@00@10.145.80.52:554", width=640, height=480),
            KafkaCommandFactory(
            bootstrap_servers=self.config.KAFKA_BOOTSTRAP_SERVERS,
            topic=self.config.KAFKA_TOPIC,
            group_id=self.config.KAFKA_GROUP_ID,
            width=self.config.PROCESSING_WIDTH,
            height=self.config.PROCESSING_HEIGHT,
            target_camera_id=_target_camera
            )
        ]

    def run(self):
        logging.info("A iniciar o sistema...")
        self._start_identity_server()
        self._start_camera_processes()
        self._wait_for_processes()
        self._shutdown()

    def _start_identity_server(self):
        """Sobe o processo dedicado que hospeda a única instância real do GlobalIdentityManager."""
        self.manager = IdentityManagerServer(authkey=b"smartbuilds-reid")
        self.manager.start()
        self.global_manager_proxy = self.manager.GlobalIdentityManager(self.config)
        logging.info("Servidor de identidades globais iniciado.")

    def _start_camera_processes(self):
        for idx, factory in enumerate(self._camera_factories()):
            proc_name = f"CameraProc-{idx}"
            p = mp.Process(
                target=run_camera_process,
                args=(factory, self.config, self.global_manager_proxy, self.stop_event, proc_name),
                name=proc_name,
            )
            p.start()
            self.processes.append(p)
            logging.info(f"Processo '{proc_name}' iniciado (pid={p.pid}).")

    def _wait_for_processes(self):
        try:
            while any(p.is_alive() for p in self.processes):
                for p in self.processes:
                    p.join(timeout=0.5)
        except KeyboardInterrupt:
            logging.info("Interrupção manual (Ctrl+C). Encerrando todos os processos...")

    def _shutdown(self):
        logging.info("A iniciar rotina de encerramento seguro...")
        self.stop_event.set()

        for p in self.processes:
            p.join(timeout=5.0)
            if p.is_alive():
                logging.warning(f"Processo '{p.name}' não encerrou a tempo, finalizando à força.")
                p.terminate()

        logging.info("A exportar dados para CSV...")
        self.global_manager_proxy.export_data_to_csv("tracking_data_final.csv")

        self.manager.shutdown()
        logging.info("Programa finalizado com sucesso.")