def __get__(self, obj, objtype=None):
    if obj is None:
        return self
    if self.fget is None:
        raise AttributeError('unreadable attribute')
    attr = '__cached_' + self.fget.__name__
    cached = getattr(obj, attr, None)
    if cached is None:
        cached = self.fget(obj)
        setattr(obj, attr, cached)
    return cached
