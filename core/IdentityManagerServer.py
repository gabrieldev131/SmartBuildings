# core/IdentityManagerServer.py
from multiprocessing.managers import BaseManager
from core.GlobalIdentityManager import GlobalIdentityManager


class IdentityManagerServer(BaseManager):
    """
    Processo dedicado que hospeda a ÚNICA instância real do
    GlobalIdentityManager.

    Os processos de câmera NÃO têm essa instância na própria memória --
    eles recebem um "proxy": um objeto local que, a cada chamada de
    método (get_or_create_global_id, update_existing_identity, ...), faz
    uma chamada por socket até este processo, que executa o método de
    verdade sobre o único dicionário de identidades que existe, e devolve
    o resultado. Do ponto de vista de quem chama, parece uma chamada de
    método normal.

    Isso resolve o problema de memória entre processos: não existem N
    cópias divergentes do estado, existe uma única fonte de verdade.
    Como efeito colateral (desejado): como todo processo de câmera passa
    pelo mesmo servidor, as decisões de Re-ID entre câmeras ficam
    automaticamente serializadas -- nunca duas câmaras decidem "essa é
    uma pessoa nova" para a mesma pessoa ao mesmo tempo. O lock que já
    existe dentro do GlobalIdentityManager (self._lock) continua
    necessário aqui: o servidor do multiprocessing.managers atende cada
    conexão recebida numa thread própria dentro do SEU processo, então
    chamadas vindas de câmeras diferentes ainda podem concorrer entre si
    dentro do processo do servidor.
    """
    pass


IdentityManagerServer.register('GlobalIdentityManager', GlobalIdentityManager)