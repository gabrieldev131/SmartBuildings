# Sistema de Monitoramento e Análise de Comportamento com Visão Computacional

## Visão Geral

Este projeto é um sistema de vigilância inteligente capaz de conectar-se a múltiplas câmeras IP via protocolo RTSP, detectar pessoas em tempo real utilizando redes neurais profundas (YOLOv3) e analisar seu comportamento (movimentação e tempo de permanência) utilizando algoritmos de fluxo óptico.

O sistema foi projetado com foco em alta performance, utilizando paralelismo (Multiprocessing) para detecção pesada e concorrência (Multithreading) para I/O de rede, garantindo fluidez e robustez.

#### Arquitetura do Sistema

O projeto segue o padrão de arquitetura MVC (Model-View-Controller) adaptado para uma aplicação de processamento de vídeo em tempo real.

Diagrama de Classes e Módulos
---
smart_builds/ <br>
├── main.py                 &nbsp;&nbsp;&nbsp;&nbsp;# Ponto de entrada <br>
├── config.py               &nbsp;&nbsp;&nbsp;&nbsp;# Configurações globais <br>
│ <br>
├── controller/ <br>
│   ├── app_controller.py   &nbsp;&nbsp;&nbsp;&nbsp;# Gerente principal do sistema <br>
│   └── camera_processor.py &nbsp;&nbsp;&nbsp;&nbsp;# Gerente de lógica por câmera <br>
│ <br>
├── model/ <br>
│   ├── network_scanner.py  &nbsp;&nbsp;&nbsp;&nbsp;# Descoberta de dispositivos <br>
│   ├── camera.py           &nbsp;&nbsp;&nbsp;&nbsp;# Leitura de vídeo (I/O) <br>
│   ├── processing_worker.py &nbsp;&nbsp;&nbsp;&nbsp;# Detecção pesada (Processos isolados) <br>
│   ├── tracker.py          &nbsp;&nbsp;&nbsp;&nbsp;# Rastreamento e Estado (Stateful) <br>
│   └── data_structures.py  &nbsp;&nbsp;&nbsp;&nbsp;# Estruturas de dados auxiliares <br>
│ <br>
└── view/ <br>
    └── console_view.py     &nbsp;&nbsp;&nbsp;&nbsp;# Interface com o usuário <br>


## Detalhamento das Classes

### 1. Camada de Controle (Controller)

#### AppController (controller/app_controller.py)

* Função: É o "maestro" da aplicação. Não realiza processamento pesado, apenas orquestra o ciclo de vida do software.

* Responsabilidades: 
* * Inicializar o sistema (_setup), incluindo threads e processos.

* * Gerenciar o loop principal (_main_loop), que distribui frames das filas para os processadores.

* * Garantir o encerramento limpo e seguro de todos os recursos (_shutdown), evitando deadlocks e processos zumbis.
---
#### CameraProcessor (controller/camera_processor.py)

* Função: O cérebro lógico de uma única câmera.

* Responsabilidades:

* * Manter o estado de uma câmera específica (contagem de frames, lista de rastreadores ativos).

* * Decidir, frame a frame, se deve enviar a imagem para detecção (YOLO) ou apenas atualizar o rastreamento (Fluxo Óptico).

* * Realizar a Associação de Dados (Data Association) usando IoU (Intersection over Union) para identificar se uma nova detecção corresponde a uma pessoa que já estava sendo rastreada.

* * Desenhar as caixas delimitadoras e informações visuais no frame final.

## 2. Camada de Modelo (Model)

#### FrameReader (model/camera.py)

* Função: O "produtor" de dados.

* Responsabilidades:

* * Conectar-se ao stream RTSP da câmera em uma thread dedicada.

* * Ler frames continuamente e colocá-los em uma fila (Queue) para serem consumidos pelo Controller.

* * Gerenciar reconexões automáticas em caso de falha de rede.

* * Redimensionar o frame na fonte para economizar banda de memória.

--- 
#### StatefulTracker (model/tracker.py)

* Função: Representa uma única pessoa sendo rastreada no tempo.

* Responsabilidades:

* * Utilizar o algoritmo Lucas-Kanade Optical Flow para estimar a nova posição da pessoa nos frames onde o YOLO não roda.

* * Manter um histórico temporal de posições.

* * Implementar a Máquina de Estados para detecção de parada ("Movendo" vs "Parado") usando lógica de histerese baseada em tempo e deslocamento.

* * Suavizar o movimento da caixa delimitadora para evitar tremores visuais.
 --- 
#### NetworkScanner (model/network_scanner.py)

* Função: Utilitário de rede.

* Responsabilidades:

* * Escanear uma faixa de IPs na rede local.

* * Identificar quais IPs possuem a porta RTSP (554) aberta, retornando a lista de câmeras disponíveis automaticamente.
---
#### processing_worker (model/processing_worker.py)

* Função: Módulo de processamento paralelo.

* Responsabilidades:

* * Este arquivo não contém uma classe, mas funções que rodam em processos separados (Multiprocessing).

* * Carrega o modelo YOLOv3 na memória de cada processo.

* * Executa a inferência da rede neural (pesada) sem bloquear o loop principal da aplicação.

#### BoundingBox e StatefulTimer (model/data_structures.py)

* Função: Classes de dados (Value Objects).

* Responsabilidades:

* * Encapsular dados primitivos para tornar o código mais legível e seguro (ex: garantir que coordenadas x, y, w, h sejam tratadas como uma entidade única).

* * StatefulTimer abstrai a lógica de cronômetros (start, reset, has_exceeded), simplificando a lógica de negócios no rastreador.

## 3. Camada de Visualização (View)

#### ConsoleView (view/console_view.py)

* Função: Interface de saída.

* Responsabilidades:

* * Exibir mensagens de log e status no terminal.

* * Gerenciar as janelas do OpenCV (cv2.imshow) para exibir o vídeo.

* * Capturar eventos de teclado (ex: tecla 'q' para sair).
---
## 4. Configuração

#### config.py

* Função: Central de configurações.

* Responsabilidades:

* * Armazenar constantes como IPs, credenciais, caminhos de arquivos de modelo, limiares de detecção (confiança, IoU) e parâmetros de comportamento (tempo para considerar "parado").

* * Fluxo de Dados Simplificado

* * Inicialização: AppController usa NetworkScanner para achar câmeras e inicia FrameReader (threads) e processing_worker (pool de processos).

* * Entrada: FrameReader lê frames da câmera e coloca na raw_frames_queue.

* * Loop Principal: AppController pega um frame da fila e o entrega ao CameraProcessor correspondente.

* Processamento:

* * Se for hora de detectar: CameraProcessor envia o frame para o processing_worker (pool). Quando volta, atualiza os rastreadores via IoU.

* * Se for hora de rastrear: CameraProcessor pede ao StatefulTracker para calcular o fluxo óptico.

Análise: StatefulTracker atualiza seu histórico e verifica se a pessoa está parada.

Saída: CameraProcessor desenha os resultados e AppController pede para a ConsoleView exibir na tela.

### Requisitos

- Python 3.10+

- OpenCV (opencv-contrib-python)

- Numpy

- Arquivos de modelo YOLOv3 (weights, cfg, names)