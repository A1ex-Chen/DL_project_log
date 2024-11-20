def add(self, module, name=None):
    if name is None:
        name = str(len(self._modules))
        if name in self._modules:
            raise KeyError('name exists')
    self.add_module(name, module)
