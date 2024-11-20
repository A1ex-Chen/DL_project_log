def __missing__(self, name):
    raise KeyError(name)
