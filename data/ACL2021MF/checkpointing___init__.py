def __init__(self, models: Union[nn.Module, Dict[str, nn.Module]],
    serialization_dir: str, mode: str='max', filename_prefix: str='model'):
    if isinstance(models, nn.Module):
        models = {'model': models}
    for key in models:
        if not isinstance(models[key], nn.Module):
            raise TypeError('{} is not a Module'.format(type(models).__name__))
    self._models = models
    self._serialization_dir = serialization_dir
    self._mode = mode
    self._filename_prefix = filename_prefix
    self._best_metric: Optional[Union[float, torch.Tensor]] = None
