def register(self, module_name, module=None):
    if module is not None:
        _register_generic(self, module_name, module)
        return

    def register_fn(fn):
        _register_generic(self, module_name, fn)
        return fn
    return register_fn
