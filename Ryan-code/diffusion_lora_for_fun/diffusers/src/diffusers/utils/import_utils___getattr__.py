def __getattr__(self, name: str) ->Any:
    if name in self._objects:
        return self._objects[name]
    if name in self._modules:
        value = self._get_module(name)
    elif name in self._class_to_module.keys():
        module = self._get_module(self._class_to_module[name])
        value = getattr(module, name)
    else:
        raise AttributeError(f'module {self.__name__} has no attribute {name}')
    setattr(self, name, value)
    return value
