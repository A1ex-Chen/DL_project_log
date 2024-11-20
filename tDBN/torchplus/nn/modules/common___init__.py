def __init__(self, *args, **kwargs):
    super(Sequential, self).__init__()
    if len(args) == 1 and isinstance(args[0], OrderedDict):
        for key, module in args[0].items():
            self.add_module(key, module)
    else:
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    for name, module in kwargs.items():
        if sys.version_info < (3, 6):
            raise ValueError('kwargs only supported in py36+')
        if name in self._modules:
            raise ValueError('name exists.')
        self.add_module(name, module)
