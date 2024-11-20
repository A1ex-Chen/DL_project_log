def build_engine(onnx_file, json_file, engine_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.max_workspace_size = GiB(1)
    if not os.path.exists(onnx_file):
        quit('ONNX file {} not found'.format(onnx_file))
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    config.set_flag(trt.BuilderFlag.INT8)
    setDynamicRange(network, json_file)
    engine = builder.build_engine(network, config)
    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())
