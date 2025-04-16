@try_export
def export_engine(model, im, file, half, dynamic, simplify, workspace=4,
    verbose=False, prefix=colorstr('TensorRT:')):
    assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('nvidia-tensorrt', cmds=
                '-U --index-url https://pypi.ngc.nvidia.com')
        import tensorrt as trt
    if trt.__version__[0] == '7':
        grid = model.model[-1].anchor_grid
        model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, im, file, 12, dynamic, simplify)
        model.model[-1].anchor_grid = grid
    else:
        check_version(trt.__version__, '8.0.0', hard=True)
        export_onnx(model, im, file, 12, dynamic, simplify)
    onnx = file.with_suffix('.onnx')
    LOGGER.info(
        f'\n{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    f = file.with_suffix('.engine')
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(
            f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(
            f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')
    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.info(
                f'{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument'
                )
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.
                shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)
    LOGGER.info(
        f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}'
        )
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    return f, None
