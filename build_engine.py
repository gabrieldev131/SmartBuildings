# build_engine.py
import tensorrt as trt


def _parse_onnx_model(builder: trt.Builder, logger: trt.Logger, onnx_path: str):
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for err in range(parser.num_errors):
                print(parser.get_error(err))
            return None
    return network


def construir_motor() -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = _parse_onnx_model(builder, logger, "models/yolo26n.onnx")
    if network is None:
        return
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.FP16)
    engine_bytes = builder.build_serialized_network(network, config)
    with open("models/yolo26n.engine", "wb") as f:
        f.write(engine_bytes)
    print("Sucesso! Motor gerado em: models/yolo26n.engine")


if __name__ == "__main__":
    construir_motor()
