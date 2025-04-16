def save(self, model: Model, model_path: Union[str, Path]) ->None:
    model_path = Path(model_path)
    LOGGER.debug(f'Saving ONNX model to {model_path.as_posix()}')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_model: onnx.ModelProto = model.handle
    if self._as_text:
        with model_path.open('w') as f:
            f.write(text_format.MessageToString(onnx_model))
    else:
        with model_path.open('wb') as f:
            f.write(onnx_model.SerializeToString())
