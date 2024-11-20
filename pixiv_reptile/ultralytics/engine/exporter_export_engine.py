@try_export
def export_engine(self, prefix=colorstr('TensorRT:')):
    """YOLOv8 TensorRT export https://developer.nvidia.com/tensorrt."""
    assert self.im.device.type != 'cpu', "export running on CPU but must be on GPU, i.e. use 'device=0'"
    f_onnx, _ = self.export_onnx()
    try:
        import tensorrt as trt
    except ImportError:
        if LINUX:
            check_requirements('tensorrt>7.0.0,<=10.1.0')
        import tensorrt as trt
    check_version(trt.__version__, '>=7.0.0', hard=True)
    check_version(trt.__version__, '<=10.1.0', msg=
        'https://github.com/ultralytics/ultralytics/pull/14239')
    LOGGER.info(
        f'\n{prefix} starting export with TensorRT {trt.__version__}...')
    is_trt10 = int(trt.__version__.split('.')[0]) >= 10
    assert Path(f_onnx).exists(), f'failed to export ONNX file: {f_onnx}'
    f = self.file.with_suffix('.engine')
    logger = trt.Logger(trt.Logger.INFO)
    if self.args.verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = int(self.args.workspace * (1 << 30))
    if is_trt10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and self.args.half
    int8 = builder.platform_has_fast_int8 and self.args.int8
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(f_onnx):
        raise RuntimeError(f'failed to load ONNX file: {f_onnx}')
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(
            f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(
            f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')
    if self.args.dynamic:
        shape = self.im.shape
        if shape[0] <= 1:
            LOGGER.warning(
                f"{prefix} WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'"
                )
        profile = builder.create_optimization_profile()
        min_shape = 1, shape[1], 32, 32
        max_shape = *shape[:2], *(max(1, self.args.workspace) * d for d in
            shape[2:])
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape
                )
        config.add_optimization_profile(profile)
    LOGGER.info(
        f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {f}"
        )
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_calibration_profile(profile)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


        class EngineCalibrator(trt.IInt8Calibrator):

            def __init__(self, dataset, batch: int, cache: str='') ->None:
                trt.IInt8Calibrator.__init__(self)
                self.dataset = dataset
                self.data_iter = iter(dataset)
                self.algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
                self.batch = batch
                self.cache = Path(cache)

            def get_algorithm(self) ->trt.CalibrationAlgoType:
                """Get the calibration algorithm to use."""
                return self.algo

            def get_batch_size(self) ->int:
                """Get the batch size to use for calibration."""
                return self.batch or 1

            def get_batch(self, names) ->list:
                """Get the next batch to use for calibration, as a list of device memory pointers."""
                try:
                    im0s = next(self.data_iter)['img'] / 255.0
                    im0s = im0s.to('cuda'
                        ) if im0s.device.type == 'cpu' else im0s
                    return [int(im0s.data_ptr())]
                except StopIteration:
                    return None

            def read_calibration_cache(self) ->bytes:
                """Use existing cache instead of calibrating again, otherwise, implicitly return None."""
                if self.cache.exists() and self.cache.suffix == '.cache':
                    return self.cache.read_bytes()

            def write_calibration_cache(self, cache) ->None:
                """Write calibration cache to disk."""
                _ = self.cache.write_bytes(cache)
        config.int8_calibrator = EngineCalibrator(dataset=self.
            get_int8_calibration_dataloader(prefix), batch=2 * self.args.
            batch, cache=str(self.file.with_suffix('.cache')))
    elif half:
        config.set_flag(trt.BuilderFlag.FP16)
    del self.model
    gc.collect()
    torch.cuda.empty_cache()
    build = (builder.build_serialized_network if is_trt10 else builder.
        build_engine)
    with build(network, config) as engine, open(f, 'wb') as t:
        meta = json.dumps(self.metadata)
        t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
        t.write(meta.encode())
        t.write(engine if is_trt10 else engine.serialize())
    return f, None
