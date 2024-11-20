def save(self, model: Model, model_path: Union[str, Path]) ->None:
    model_path = Path(model_path)
    LOGGER.debug(f'Saving TensorRT engine to {model_path.as_posix()}')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    engine: 'trt.ICudaEngine' = model.handle
    with model_path.open('wb') as fh:
        fh.write(engine.serialize())
