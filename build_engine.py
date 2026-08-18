import tensorrt as trt

def construir_motor():
    print("Iniciando a conversão direta de ONNX para TensorRT...")
    
    # Configuração do Logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    
    # No TensorRT 10+, o EXPLICIT_BATCH já é o padrão
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Lendo o arquivo ONNX que você já tem
    onnx_path = "models/yolo26n.onnx"
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Erro ao ler o ONNX. Verifique os logs do TensorRT.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
            
    print("ONNX carregado! Otimizando matrizes para a sua RTX 2080 Ti...")
    config = builder.create_builder_config()
    
    # Liberando até 4GB de VRAM para o compilador achar a melhor rota matemática
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    
    # Forçando precisão FP16 (metade da memória, dobro da velocidade)
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Construindo o motor
    engine_bytes = builder.build_serialized_network(network, config)
    
    # Salvando no disco
    engine_path = "models/yolo26n.engine"
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
        
    print(f"Sucesso! Motor gerado em: {engine_path}")

if __name__ == "__main__":
    construir_motor()