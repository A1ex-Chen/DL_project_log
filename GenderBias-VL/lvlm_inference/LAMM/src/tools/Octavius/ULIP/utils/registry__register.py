def _register(cls):
    self._register_module(module_class=cls, module_name=name, force=force)
    return cls
