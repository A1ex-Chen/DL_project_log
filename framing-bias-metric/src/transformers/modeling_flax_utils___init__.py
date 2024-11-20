def __init__(self, config: PretrainedConfig, module: nn.Module, params:
    Dict, seed: int=0):
    if config is None:
        raise ValueError('config cannot be None')
    if module is None:
        raise ValueError('module cannot be None')
    if params is None:
        raise ValueError('state cannot be None')
    self._config = config
    self._module = module
    self.key = PRNGKey(seed)
    self.params = params
    self.model = module
