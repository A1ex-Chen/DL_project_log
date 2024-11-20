def __init__(self, model: Model, providers, verbose_runtime_logs: bool=False):
    super().__init__(model)
    self._input_names = None
    self._output_names = None
    self._session = None
    self._providers = providers
    self._verbose_runtime_logs = verbose_runtime_logs
    self._old_env_values = {}
