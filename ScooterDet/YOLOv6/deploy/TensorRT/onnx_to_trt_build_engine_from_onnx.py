def build_engine_from_onnx(model_name, dtype, verbose=False, int8_calib=
    False, calib_loader=None, calib_cache=None, fp32_layer_names=[],
    fp16_layer_names=[]):
    """Initialization routine."""
    if dtype == 'int8':
        t_dtype = trt.DataType.INT8
    elif dtype == 'fp16':
        t_dtype = trt.DataType.HALF
    elif dtype == 'fp32':
        t_dtype = trt.DataType.FLOAT
    else:
        raise ValueError('Unsupported data type: %s' % dtype)
    if trt.__version__[0] < '8':
        print('Exit, trt.version should be >=8. Now your trt version is ',
            trt.__version__[0])
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if dtype == 'int8' and calib_loader is None:
        print('QAT enabled!')
        network_flags = network_flags | 1 << int(trt.
            NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    """Build a TensorRT engine from ONNX"""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=
        network_flags) as network, trt.OnnxParser(network, TRT_LOGGER
        ) as parser:
        with open(model_name, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: ONNX Parse Failed')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                    return None
        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" or "-v" to enable verbose logging.)')
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30
        if t_dtype == trt.DataType.HALF:
            config.flags |= 1 << int(trt.BuilderFlag.FP16)
        if t_dtype == trt.DataType.INT8:
            print('trt.DataType.INT8')
            config.flags |= 1 << int(trt.BuilderFlag.INT8)
            config.flags |= 1 << int(trt.BuilderFlag.FP16)
            if int8_calib:
                from calibrator import Calibrator
                config.int8_calibrator = Calibrator(calib_loader, calib_cache)
                print('Int8 calibation is enabled.')
        engine = builder.build_engine(network, config)
        try:
            assert engine
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError('Parsing failed on line {} in statement {}'
                .format(line, text))
        return engine
