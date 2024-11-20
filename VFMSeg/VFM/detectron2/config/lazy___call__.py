def __call__(self, **kwargs):
    if is_dataclass(self._target):
        target = _convert_target_to_string(self._target)
    else:
        target = self._target
    kwargs['_target_'] = target
    return DictConfig(content=kwargs, flags={'allow_objects': True})
