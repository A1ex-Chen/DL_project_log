def __init__(self, model: Model):
    super().__init__(model)
    assert isinstance(model.handle, trt.ICudaEngine)
    self._model = model
    self._has_dynamic_shapes = None
    self._context = None
    self._engine: trt.ICudaEngine = self._model.handle
    self._cuda_context = pycuda.autoinit.context
    self._input_names = None
    self._output_names = None
    self._buffers = None
