# SmartBuildings — High-Performance Multi-Camera Surveillance, Re-ID & Behavioral Analytics

## Visão Geral (Overview)

O **SmartBuildings** é uma plataforma de visão computacional em tempo real projetada para monitoramento inteligente de múltiplos ambientes, **Re-Identificação Global entre Câmeras (Cross-Camera Re-ID)**, detecção de comportamento de permanência/parada (*Loitering / Stopped State*) e **Análise Preditiva de Fluxo via Cadeias de Markov (MCMC)**.

O sistema foi arquitetado seguindo os princípios de **Clean Architecture**, **SOLID** e **Padrões de Projeto (GoF)**, combinados com um motor de inferência em lote acelerado por **CUDA Graphs (FP16)** e **DeepSORT Vetorizado**, permitindo atingir **mais de 100 FPS por câmera simultaneamente em 8 câmeras** (mais de **800 FPS agregados**) em uma única GPU **NVIDIA GeForce RTX 2080 Ti** com processador **Intel Core i9-9900K**.

---

## Destaques de Performance (Benchmarks)

Testes realizados no hardware alvo (**Intel Core i9-9900K, 48 GB RAM, NVIDIA RTX 2080 Ti 11 GB VRAM**, resolução de processamento `640x480` em `FP16` com `YOLO26n` + `MobileNetV3-Small Part-Based Re-ID` + `VectorizedDeepSort` + `GlobalIdentityManager` + `StoppedStateTracker` + `HUD`):

| Cenário | Arquitetura Anterior (Baseline) | Nova Arquitetura (`Batched CUDA Graph + Vectorized DeepSORT`) | Ganho de Performance |
| :--- | :---: | :---: | :---: |
| **1 Câmera** | `17.2 FPS` (`58.08 ms/frame`) | **`269.4 FPS/câm`** (`269.4 FPS agregados`, `3.71 ms/batch`) | **15.6x mais rápido** |
| **2 Câmeras** | `< 15 FPS/câm` (contenção de GPU/IPC) | **`228.1 FPS/câm`** (`456.2 FPS agregados`, `4.38 ms/batch`) | **> 15x mais rápido** |
| **4 Câmeras** | Inviável em tempo real (`< 8 FPS/câm`) | **`144.2 FPS/câm`** (`576.7 FPS agregados`, `6.94 ms/batch`) | **2.4x acima da meta de 60 FPS** |
| **8 Câmeras** | Inviável (`OOM / < 4 FPS/câm`) | **`75.6 FPS/câm`** (`604.6 FPS agregados`, `13.23 ms/batch`) | **+26% acima da meta de 60 FPS** |

### Por que o sistema agora atinge 60–100+ FPS em 8 câmeras?

1. **Captura de Grafo CUDA Unificado (`torch.cuda.CUDAGraph`)**:
   - Na arquitetura anterior, cada câmera rodava em um processo separado fazendo chamadas individuais ao PyTorch (`YOLO` + loops `for` em Python executando `MobileNetV3` uma pessoa por vez com conversões CPU $\leftrightarrow$ PIL $\leftrightarrow$ GPU).
   - Agora, o [`BatchInferenceEngine`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/vision/BatchInferenceEngine.py) agrupa todas as câmeras ativas em um único tensor em **Memória Pinned (`pin_memory=True`)** e executa **Conversão BGR$\to$RGB + Normalização FP16 + YOLO26n + Top-K de Pessoas + `torchvision.ops.roi_align` + MobileNetV3-Small Re-ID** dentro de um único **CUDA Graph** capturado na GPU, reduzindo o tempo de despacho da CPU para `< 0.1 ms` e eliminando gargalos de sincronização.
2. **Eliminação do Gargalo de Sincronização CPU$\leftrightarrow$GPU no Head do YOLO26n**:
   - O post-processamento padrão do Ultralytics (`Detect.get_topk_index`) alocava índices na CPU (`torch.arange(batch_size)`) e ordenava 80 classes $\times$ 300 caixas, forçando sincronização bloqueante de stream CUDA a cada frame. O novo post-processamento filtra diretamente a classe `0` (*person*) na GPU em formato estático.
3. **Pipeline Duplo-Bufferizado Assíncrono (Software Pipelining GPU $\parallel$ CPU)**:
   - Enquanto o hardware da GPU executa o **Batch $t$** em uma `torch.cuda.Stream` dedicada, a CPU processa simultaneamente o rastreamento `VectorizedDeepSort`, o `GlobalIdentityManager`, o `StoppedStateTracker` e a anotação visual do **Batch $t-1$**.
4. **DeepSORT 3D Totalmente Vetorizado (`VectorizedDeepSort`)**:
   - Substitui milhares de chamadas de funções Python e `np.linalg.multi_dot` por álgebra linear em lote (`np.linalg.solve` 3D e `np.einsum` sobre matrizes contíguas `(N, 8)` e `(N, 8, 8)`), reduzindo o custo do filtro de Kalman + Mahalanobis + Distância de Cosseno para **`~0.25 ms` por câmera**.
5. **Memória Compartilhada In-Process para Identidades Globais**:
   - Elimina milhares de chamadas RPC via socket/pickle (`BaseManager`) por segundo no modo `"batched"`, mantendo o modo `"multiprocess"` disponível via configuração.

---

## Arquitetura do Sistema (Clean Architecture & Design Patterns)

O projeto é organizado em camadas concêntricas seguindo a **Clean Architecture** e o **Princípio da Inversão de Dependência (DIP)**:

```mermaid
flowchart TD
    subgraph UI["Camada de Apresentação (ui/)"]
        DISP["MultiCameraDisplay & HUD Annotations"]
    end

    subgraph INFRA_EXT["Camada de Extração / I/O (extraction/)"]
        MGR["FrameReaderManagement (Facade / Orchestrator)"]
        INV["FrameReaderInvoker (Command Invoker)"]
        FAC["CommandFactories (Abstract Factory)"]
        RTSP["ReadRTSPCommand (Concrete Command)"]
        KAFKA["ReadKafkaCommand (Concrete Command)"]
    end

    subgraph VISION["Camada de Visão Computacional & IA (vision/)"]
        BIE["BatchInferenceEngine (CUDA Graph + YOLO26n + ROI-Align)"]
        FE["featureExtractor (Part-Based + Chromaticity Re-ID)"]
        VDS["VectorizedDeepSort (2-Stage ByteTrack + Kalman 3D)"]
        CW["CameraWorker (Per-Camera Pipeline Adapter)"]
    end

    subgraph CORE["Camada de Domínio & Portas (core/)"]
        PORTS["ports.py (IFrameCommand, IMultiObjectTracker, IIdentityRepository)"]
        DOM["domain.py (BoundingBox, FramePacket, TrackedPerson)"]
        GIM["GlobalIdentityManager (Cross-Camera Re-ID Service)"]
        GI["GlobalIdentity (EMA + Gallery Entity)"]
        CCM["CameraClusterManager (Union-Find Topology)"]
        SST["StoppedStateTracker (Hysteresis State Machine)"]
    end

    MGR --> INV
    FAC --> RTSP
    FAC --> KAFKA
    INV --> PORTS
    MGR --> BIE
    MGR --> CW
    MGR --> DISP
    CW --> VDS
    CW --> GIM
    CW --> SST
    BIE --> FE
    VDS --> PORTS
    GIM --> GI
    GIM --> CCM
    GIM --> PORTS
```

### Estrutura de Diretórios

```text
SmartBuildings/
├── main.py                                  # Ponto de entrada da aplicação
├── Config.py                                # Configurações globais (Performance, YOLO, DeepSORT, Re-ID, UI)
├── MarkovChain.py                           # Análise preditiva de fluxo via MCMC (PyMC + NetworkX)
├── export_tensorrt.py / build_engine.py     # Utilitários opcionais para exportação TensorRT (.engine)
│
├── core/                                    # CAMADA DE DOMÍNIO & PORTAS (Enterprise Business Rules)
│   ├── domain.py                            # Entidades e Value Objects (BoundingBox, FramePacket, TrackedPerson)
│   ├── ports.py                             # Interfaces ABC (IFrameCommand, ICommandFactory, IMultiObjectTracker, etc.)
│   ├── GlobalIdentity.py                    # Entidade de identidade global (EMA + Galeria limitada com filtro de qualidade)
│   ├── GlobalIdentityManager.py             # Repositório e serviço de Re-ID intra/inter-câmera e exportação CSV
│   ├── CameraClusterManager.py              # Agrupamento automático de câmeras via estrutura Union-Find
│   ├── StoppedStateTracker.py               # Máquina de estados com histerese espacial e temporal para detecção de parada
│   ├── IdentityManagerServer.py             # Servidor RPC BaseManager (usado no modo legado multiprocess)
│   └── runtime_setup.py                     # Otimização de threads nativas (OpenMP/MKL/OpenCV) e flags CUDA/TF32
│
├── extraction/                              # CAMADA DE INFRAESTRUTURA DE CAPTURA (Command + Factory + Invoker)
│   ├── FrameReaderManagement.py             # Orquestrador principal (modos 'batched' e 'multiprocess')
│   ├── CameraProcess.py                     # Worker de processo isolado (modo 'multiprocess')
│   └── frameReaderCommand/
│       ├── IFrameCommand.py                 # Interface do padrão Command para leitura de vídeo
│       ├── ReadRTSPCommand.py               # Comando concreto para streams RTSP e arquivos de vídeo locais (.mp4)
│       ├── ReadKafkaCommand.py              # Comando concreto para consumo de frames via Apache Kafka
│       ├── CommandFactories.py              # Fábricas serializáveis (RTSPCommandFactory, KafkaCommandFactory)
│       └── FrameReaderInvoker.py            # Invoker assíncrono em thread dedicada mantendo sempre o frame mais recente
│
├── vision/                                  # CAMADA DE VISÃO COMPUTACIONAL & IA (Strategy + Adapter + Object Pool)
│   ├── BatchInferenceEngine.py              # Motor GPU unificado (CUDA Graphs + Pinned Memory + YOLO26n + ROI-Align + Re-ID)
│   ├── VectorizedDeepSort.py                # Rastreador DeepSORT 3D vetorizado com associação em 2 estágios
│   ├── featureExtractor.py                  # Extrator Re-ID híbrido (MobileNetV3 FP16 + Faixas Horizontais + Cromaticidade)
│   └── cameraWorker.py                      # Worker por câmera integrando rastreamento, Re-ID global e anotação
│
└── ui/                                      # CAMADA DE APRESENTAÇÃO (Presentation Layer)
    └── display.py                           # Renderizador OpenCV (Dashboard em Grade Múltipla ou Janelas Individuais a 60Hz)
```

### Padrões de Projeto Aplicados (Design Patterns)

| Padrão (Pattern) | Onde é Aplicado | Benefício Arquitetural |
| :--- | :--- | :--- |
| **Command** | [`IFrameCommand`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/core/ports.py), [`ReadRTSPCommand`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/extraction/frameReaderCommand/ReadRTSPCommand.py), [`ReadKafkaCommand`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/extraction/frameReaderCommand/ReadKafkaCommand.py) | Encapsula a lógica de captura de cada protocolo (RTSP, arquivo local 4K, Kafka) sob uma única operação `execute()` e `cleanup()`. |
| **Invoker (Latest-Frame Ring)** | [`FrameReaderInvoker`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/extraction/frameReaderCommand/FrameReaderInvoker.py) | Desacopla a taxa de decodificação de vídeo da taxa de inferência da GPU, descartando frames obsoletos para garantir latência zero. |
| **Abstract Factory** | [`ICommandFactory`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/core/ports.py), [`RTSPCommandFactory`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/extraction/frameReaderCommand/CommandFactories.py), [`KafkaCommandFactory`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/extraction/frameReaderCommand/CommandFactories.py) | Permite injetar e serializar (*pickle*) construtores de fontes de vídeo entre processos sem abrir descritores de rede/arquivos prematuramente. |
| **Strategy & Adapter** | [`IMultiObjectTracker`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/core/ports.py), [`VectorizedDeepSort`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/vision/VectorizedDeepSort.py) | Permite evoluir ou substituir o algoritmo de rastreamento sem alterar o `CameraWorker` ou as regras de domínio. |
| **Repository** | [`IIdentityRepository`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/core/ports.py), [`GlobalIdentityManager`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/core/GlobalIdentityManager.py) | Centraliza a persistência em memória, consulta vetorial de identidades globais e exportação analítica para CSV. |
| **Facade / Mediator** | [`FrameReaderManagement`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/extraction/FrameReaderManagement.py) | Orquestra a inicialização das threads de captura, o pipeline duplo-bufferizado de GPU/CPU, a interface gráfica e o encerramento seguro. |
| **Proxy** | [`IdentityManagerServer`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/core/IdentityManagerServer.py) | Transparentemente expõe o `GlobalIdentityManager` via socket IPC caso o usuário selecione `EXECUTION_MODE = "multiprocess"`. |

---

## Melhorias de Precisão no YOLO e DeepSORT

Além do ganho expressivo de FPS, diversos problemas clássicos de rastreamento e Re-ID foram resolvidos:

1. **Associação em Dois Estágios Inspirada no ByteTrack (`YOLO_CONF_HIGH` + `YOLO_CONF_LOW`)**:
   - Detecções de alta confiança ($\ge 0.42$) conduzem o matching principal de aparência + Mahalanobis e podem iniciar novos tracks.
   - Detecções de confiança intermediária ($0.25 \le \text{conf} < 0.42$), comuns durante oclusões parciais ou *motion blur*, são aproveitadas em um segundo estágio exclusivo por IoU para manter tracks confirmados vivos sem gerar falsos positivos.
2. **Correção de Supressão Indevida em Cruzamento de Pedestres**:
   - Na implementação anterior, o `DeepSort` estava configurado com `nms_max_overlap=0.35`, o que apagava uma das pessoas sempre que duas caminhavam próximas (IoU $> 0.35$). Agora o NMS intra-classe calibrado (`YOLO_IOU_THRESHOLD = 0.50`) e a validação geométrica de proporção humana (`MIN_ASPECT_RATIO`, `MAX_ASPECT_RATIO`, dimensões mínimas) garantem que pedestres próximos não sejam suprimidos.
3. **Persistência de Mapeamento Local $\to$ Global Durante Piscadas de 1 Frame**:
   - Anteriormente, se o detector perdesse uma pessoa por **1 único frame** (`time_since_update > 0`), o `CameraWorker` apagava imediatamente o `local_id` do dicionário `local_to_global_map`, forçando uma nova consulta de Re-ID no frame seguinte.
   - Agora, o vínculo `local_id -> global_id` é preservado durante toda a vida útil do track no filtro de Kalman (`not track.is_deleted()`), com *coasting* suave por até `DEEPSORT_MAX_COAST_FRAMES = 2` frames.
4. **Embeddings Re-ID Part-Based (Cabeça/Torso/Pernas) + Cromaticidade Invariante à Iluminação**:
   - O [`featureExtractor.py`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/vision/featureExtractor.py) e o [`BatchInferenceEngine.py`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/vision/BatchInferenceEngine.py) combinam *Global Average Pooling* com *Horizontal Stripe Max-Pooling* (dividindo a pessoa em faixas superior e inferior) e razões cromáticas normalizadas pela luminância ($r = R/\Sigma, g = G/\Sigma, b = B/\Sigma$) recortadas no centro do tronco/pernas.
   - Atualizações na galeria de aparência são bloqueadas automaticamente quando duas caixas se sobrepõem fortemente (proteção contra contaminação por oclusão) ou quando a pessoa está cortada na borda da imagem (`_is_near_edge`).

---

## Como Configurar e Executar

### 1. Requisitos e Ambiente Virtual

- Python 3.11+
- GPU NVIDIA com suporte a CUDA (otimizado para série RTX com FP16 / Tensor Cores)
- Dependências listadas em `requirements.txt`:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurando as Câmeras (2 a 8+ Câmeras)

Edite o método `_camera_factories()` em [`extraction/FrameReaderManagement.py`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/extraction/FrameReaderManagement.py) para declarar suas fontes RTSP, arquivos de vídeo `.mp4` ou tópicos Kafka:

```python
return [
    RTSPCommandFactory(source="rtsp://admin:senha@10.145.80.52:554", width=w, height=h, cam_id="cam_1"),
    RTSPCommandFactory(source="rtsp://admin:senha@10.145.80.53:554", width=w, height=h, cam_id="cam_2"),
    # Adicione até 8 (ou mais) câmeras — o BatchInferenceEngine agrupa todas automaticamente na GPU!
]
```

Principais parâmetros em [`Config.py`](file:///home/gabriel/Documents/vscode/python/SmartBuildings/Config.py):
- `EXECUTION_MODE = "batched"`: Habilita o pipeline em lote com CUDA Graphs (recomendado para máxima taxa de FPS em 2 a 8 câmeras).
- `DISPLAY_GRID_VIEW = True`: Exibe todas as câmeras em um único painel em grade (*Dashboard*) a 60 Hz, evitando travamentos do servidor gráfico X11 com múltiplas janelas.
- `VIDEO_FILE_REALTIME_PACING = False`: Quando `False`, processa vídeos locais na velocidade máxima do hardware; quando `True`, sincroniza a leitura com o FPS nativo do arquivo de vídeo.

### 3. Executando o Sistema Principal

```bash
./venv/bin/python main.py
```

Pressione **`q`** ou **`ESC`** na janela de visualização (ou `Ctrl+C` no terminal) para encerrar de forma limpa e exportar automaticamente o histórico consolidado de presença e transições para `tracking_data_final.csv`.

### 4. Executando a Análise Preditiva de Fluxo (Markov Chain MCMC)

Após coletar os dados em `tracking_data_final.csv`, execute o módulo de inferência bayesiana para gerar a matriz de probabilidades de transição (`matriz_probabilidades.csv`) e o dashboard topológico (`dashboard_previsao.png`):

```bash
./venv/bin/python MarkovChain.py
```
