def __init__(self, target):
    if not (callable(target) or isinstance(target, (str, abc.Mapping))):
        raise TypeError(
            f'target of LazyCall must be a callable or defines a callable! Got {target}'
            )
    self._target = target
