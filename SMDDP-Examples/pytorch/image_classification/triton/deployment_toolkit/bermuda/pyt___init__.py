def __init__(self, model: Model):
    super().__init__(model)
    assert isinstance(model.handle, torch.jit.ScriptModule) or isinstance(model
        .handle, torch.nn.Module
        ), "The model must be of type 'torch.jit.ScriptModule' or 'torch.nn.Module'. Runner aborted."
    self._model = model
    self._output_names = None
