def onnx2trt(onnx_model: onnx.ModelProto, *, shapes: Dict[str, ShapeSpec],
    max_workspace_size: int, max_batch_size: int, model_precision: str
    ) ->'trt.ICudaEngine':
    """
    Converts onnx model to TensorRT ICudaEngine
    Args:
        onnx_model: onnx.Model to convert
        shapes: dictionary containing min shape, max shape, opt shape for each input name
        max_workspace_size: The maximum GPU temporary memory which the CudaEngine can use at execution time.
        max_batch_size: The maximum batch size which can be used at execution time,
                        and also the batch size for which the CudaEngine will be optimized.
        model_precision: precision of kernels (possible values: fp16, fp32)

    Returns: TensorRT ICudaEngine
    """
    fp16_mode = '16' in model_precision
    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = fp16_mode
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace_size
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        if not parser.parse(onnx_model.SerializeToString()):
            for i in range(parser.num_errors):
                LOGGER.error(
                    f'OnnxParser error {i}/{parser.num_errors}: {parser.get_error(i)}'
                    )
            raise RuntimeError(
                'Error during parsing ONNX model (see logs for details)')
        if fp16_mode:
            network.get_input(0).dtype = trt.DataType.HALF
            network.get_output(0).dtype = trt.DataType.HALF
        config = builder.create_builder_config()
        config.flags |= bool(fp16_mode) << int(trt.BuilderFlag.FP16)
        config.max_workspace_size = max_workspace_size
        profile = builder.create_optimization_profile()
        for name, spec in shapes.items():
            profile.set_shape(name, **spec._asdict())
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config=config)
    return engine
