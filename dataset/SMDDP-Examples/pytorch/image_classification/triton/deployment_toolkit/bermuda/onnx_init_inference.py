def init_inference(self, model: Model):
    assert isinstance(model.handle, onnx.ModelProto)
    return OnnxRunnerSession(model=model, providers=self._providers,
        verbose_runtime_logs=self._verbose_runtime_logs)
