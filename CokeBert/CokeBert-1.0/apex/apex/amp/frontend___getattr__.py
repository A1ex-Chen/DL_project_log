def __getattr__(self, name):
    if 'options' in self.__dict__:
        options = self.__dict__['options']
        if name in options:
            return options[name]
    raise AttributeError("'{}' object has no attribute '{}'".format(type(
        self).__name__, name))
