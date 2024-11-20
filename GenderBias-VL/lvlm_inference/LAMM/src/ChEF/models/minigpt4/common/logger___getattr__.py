def __getattr__(self, attr):
    if attr in self.meters:
        return self.meters[attr]
    if attr in self.__dict__:
        return self.__dict__[attr]
    raise AttributeError("'{}' object has no attribute '{}'".format(type(
        self).__name__, attr))
