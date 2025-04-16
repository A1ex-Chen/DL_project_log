def __call__(self):
    r = super().__call__()
    if hasattr(r, '_fix_weakref'):
        r._fix_weakref()
    return r
