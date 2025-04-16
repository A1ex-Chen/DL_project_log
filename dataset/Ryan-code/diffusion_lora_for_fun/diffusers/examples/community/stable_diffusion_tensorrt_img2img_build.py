def build(self, onnx_path, fp16, input_profile=None, enable_preview=False,
    enable_all_tactics=False, timing_cache=None, workspace_size=0):
    logger.warning(
        f'Building TensorRT engine for {onnx_path}: {self.engine_path}')
    p = Profile()
    if input_profile:
        for name, dims in input_profile.items():
            assert len(dims) == 3
            p.add(name, min=dims[0], opt=dims[1], max=dims[2])
    config_kwargs = {}
    config_kwargs['preview_features'] = [trt.PreviewFeature.
        DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
    if enable_preview:
        config_kwargs['preview_features'].append(trt.PreviewFeature.
            FASTER_DYNAMIC_SHAPES_0805)
    if workspace_size > 0:
        config_kwargs['memory_pool_limits'] = {trt.MemoryPoolType.WORKSPACE:
            workspace_size}
    if not enable_all_tactics:
        config_kwargs['tactic_sources'] = []
    engine = engine_from_network(network_from_onnx_path(onnx_path, flags=[
        trt.OnnxParserFlag.NATIVE_INSTANCENORM]), config=CreateConfig(fp16=
        fp16, profiles=[p], load_timing_cache=timing_cache, **config_kwargs
        ), save_timing_cache=timing_cache)
    save_engine(engine, path=self.engine_path)
