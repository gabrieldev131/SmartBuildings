# SmartBuildings — Multi-Camera Surveillance, Re-ID & Behavioral Analytics

## Visão Geral (Overview)

O **SmartBuildings** é uma plataforma de visão computacional em tempo real para monitoramento de múltiplos ambientes, **Re-Identificação Global entre Câmeras (Cross-Camera Re-ID)**, detecção de comportamento de parada (*Stopped State*) e **Análise Preditiva de Fluxo via Cadeias de Markov (MCMC)**.

O código foi estruturado de forma limpa e legível utilizando bibliotecas consolidadas do ecossistema Python:
- **Detecção de Pessoas**: [`ultralytics.YOLO`](https://docs.ultralytics.com/) (`YOLO26n` com inferência em lote em GPU FP16 ou CPU FP32).
- **Rastreamento Multi-Objeto (MOT)**: [`deep-sort-realtime`](https://pypi.org/project/deep-sort-realtime/) (`DeepSort` com Filtro de Kalman, distância de Mahalanobis e Hungarian Algorithm).
- **Extração de Features Re-ID em Lote**: [`torchvision`](https://pytorch.org/vision/stable/) (`models.mobilenet_v3_small` + `ops.roi_align` + `transforms.v2.functional.normalize`).
- **Matching Vetorial & Agrupamento de Câmeras**: [`scipy`](https://scipy.org/) (`scipy.spatial.distance.cdist` para distância de cosseno e `scipy.cluster.hierarchy.DisjointSet` para Union-Find de câmeras).
- **Streaming & I/O**: `opencv-python` (RTSP / MP4) e `confluent-kafka` (Apache Kafka).

---

## Estrutura de Diretórios

```text
SmartBuildings/
├── main.py                                  # Ponto de entrada da aplicação
├── Config.py                                # Configurações centralizadas (Hardware, YOLO, DeepSort, Re-ID, UI, Kafka)
├── MarkovChain.py                           # Análise preditiva de fluxo via MCMC (PyMC + NetworkX + Seaborn)
│
├── core/                                    # REGRAS DE NEGÓCIO & DOMÍNIO
│   ├── domain.py                            # Dataclasses de domínio (BoundingBox, FramePacket, TrackedPerson)
│   ├── ports.py                             # Interfaces ABC (IFrameCommand, ICommandFactory, IIdentityRepository)
│   ├── GlobalIdentity.py                    # Entidade de identidade global (deque + scipy.spatial.distance.cdist)
│   ├── GlobalIdentityManager.py             # Serviço de Re-ID intra/inter-câmera e exportação CSV
│   ├── CameraClusterManager.py              # Agrupamento automático de câmeras via scipy DisjointSet
│   ├── StoppedStateTracker.py               # Máquina de estados para detecção de pessoas paradas
│   ├── IdentityManagerServer.py             # Servidor RPC BaseManager (modo multiprocess)
│   └── runtime_setup.py                     # Configuração de threads nativas (OpenCV/PyTorch) e flags CUDA
│
├── extraction/                              # CAPTURA DE VÍDEO ASSÍNCRONA
│   ├── FrameReaderManagement.py             # Orquestrador principal (modos 'batched' e 'multiprocess')
│   ├── CameraProcess.py                     # Worker de processo isolado (modo 'multiprocess')
│   └── frameReaderCommand/
│       ├── IFrameCommand.py                 # Interface Command para leitura de vídeo
│       ├── ReadRTSPCommand.py               # Leitura de streams RTSP e vídeos locais (.mp4)
│       ├── ReadKafkaCommand.py              # Consumo de frames via Apache Kafka
│       ├── CommandFactories.py              # Fábricas de comandos (RTSPCommandFactory, KafkaCommandFactory)
│       └── FrameReaderInvoker.py            # Thread assíncrona mantendo sempre o frame mais recente
│
├── vision/                                  # VISÃO COMPUTACIONAL & IA
│   ├── BatchInferenceEngine.py              # Inferência em lote com Ultralytics YOLO + Torchvision ROI-Align
│   ├── featureExtractor.py                  # Extrator Re-ID com Torchvision MobileNetV3-Small
│   └── cameraWorker.py                      # Worker por câmera usando a biblioteca `deep-sort-realtime`
│
└── ui/                                      # APRESENTAÇÃO & HUD
    └── display.py                           # Renderização OpenCV (janelas individuais ou grade unificada)
```

---

## Como Executar

### 1. Instalar Dependências

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Fontes de Vídeo e GPU/CPU

- Em [`Config.py`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/Config.py):
  - `USE_GPU = True`: Usa GPU CUDA (FP16) automaticamente quando disponível, ou `False` para forçar CPU.
  - `EXECUTION_MODE = "batched"`: Processa múltiplas câmeras em lote na GPU.
- Em [`extraction/FrameReaderManagement.py`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/extraction/FrameReaderManagement.py), configure `_camera_factories()` com suas fontes `RTSPCommandFactory` ou `KafkaCommandFactory`.

### 3. Iniciar o Sistema

```bash
./venv/bin/python main.py
```

Pressione **`q`** ou **`ESC`** na janela de vídeo (ou `Ctrl+C` no terminal) para encerrar e gerar automaticamente o ficheiro `tracking_data_final.csv`.

### 4. Análise Preditiva (Cadeia de Markov)

```bash
./venv/bin/python MarkovChain.py
```
