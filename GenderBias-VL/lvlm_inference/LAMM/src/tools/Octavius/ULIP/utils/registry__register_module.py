def _register_module(self, module_class, module_name=None, force=False):
    if not inspect.isclass(module_class):
        raise TypeError(f'module must be a class, but got {type(module_class)}'
            )
    if module_name is None:
        module_name = module_class.__name__
    if isinstance(module_name, str):
        module_name = [module_name]
    for name in module_name:
        if not force and name in self._module_dict:
            raise KeyError(f'{name} is already registered in {self.name}')
        self._module_dict[name] = module_class
