def __enter__(self):
    self._old_env_values = self._set_env_variables()
    sess_options = onnxruntime.SessionOptions()
    if self._verbose_runtime_logs:
        sess_options.log_severity_level = 0
        sess_options.log_verbosity_level = 1
    LOGGER.info(
        f'Starting inference session for onnx model providers={self._providers} sess_options={sess_options}'
        )
    self._input_names = list(self._model.inputs)
    self._output_names = list(self._model.outputs)
    model_payload = self._model.handle.SerializeToString()
    self._session = onnxruntime.InferenceSession(model_payload, providers=
        self._providers, sess_options=sess_options)
    return self
