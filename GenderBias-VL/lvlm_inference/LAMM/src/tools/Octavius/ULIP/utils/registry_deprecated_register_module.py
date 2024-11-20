def deprecated_register_module(self, cls=None, force=False):
    warnings.warn(
        'The old API of register_module(module, force=False) is deprecated and will be removed, please use the new API register_module(name=None, force=False, module=None) instead.'
        )
    if cls is None:
        return partial(self.deprecated_register_module, force=force)
    self._register_module(cls, force=force)
    return cls
