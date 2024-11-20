def load(self, model_path: Union[str, Path], **_) ->Model:
    model_path = Path(model_path)
    LOGGER.debug(f'Loading TensorRT engine from {model_path}')
    with model_path.open('rb') as fh, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(fh.read())
    if engine is None:
        raise RuntimeError(f'Could not load ICudaEngine from {model_path}')
    inputs = {}
    outputs = {}
    for binding_idx in range(engine.num_bindings):
        name = engine.get_binding_name(binding_idx)
        is_input = engine.binding_is_input(binding_idx)
        dtype = engine.get_binding_dtype(binding_idx)
        shape = engine.get_binding_shape(binding_idx)
        if is_input:
            inputs[name] = TensorSpec(name, dtype, shape)
        else:
            outputs[name] = TensorSpec(name, dtype, shape)
    return Model(engine, None, inputs, outputs)
